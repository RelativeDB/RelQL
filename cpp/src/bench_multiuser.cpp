// bench_multiuser.cpp — concurrent-request benchmark for rt.cpp.
//
//   ./rt_bench_mu <model.safetensors> [--device cpu|mps|cuda] [--seq S]
//                 [--requests N] [--users "1,2,4,8,16"]
//
// Simulates N independent users each issuing single-entity forwards
// (context + query, batch of 1) as fast as the engine returns them, the
// serving shape the context+query server will see. Every request uses its
// own synthetic batch so host-side prepare() work is not amortized away.
// Reports aggregate throughput and per-request latency percentiles per
// concurrency level.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "bench_synth.hpp"
#include "rt.hpp"

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0).count();
}

double pct(std::vector<double>& v, double p) {
  size_t i = (size_t)(p * (v.size() - 1));
  std::nth_element(v.begin(), v.begin() + i, v.end());
  return v[i];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
            "usage: %s <safetensors> [--device cpu|mps|cuda] [--seq S] "
            "[--requests N] [--users \"1,2,4,8\"]\n", argv[0]);
    return 2;
  }
  rt::ForwardOpts opts;
  opts.debug_taps = false;
  int seq = 2048, requests = 16;
  std::vector<int> users = {1, 2, 4, 8, 16};
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string a = argv[i], v = argv[i + 1];
    if (a == "--device")
      opts.device = v == "mps" ? rt::Device::MPS
                    : v == "cuda" ? rt::Device::CUDA
                                  : rt::Device::CPU;
    else if (a == "--seq") seq = std::stoi(v);
    else if (a == "--requests") requests = std::stoi(v);
    else if (a == "--users") {
      users.clear();
      for (size_t p = 0; p < v.size();) {
        size_t q = v.find(',', p);
        if (q == std::string::npos) q = v.size();
        users.push_back(std::stoi(v.substr(p, q - p)));
        p = q + 1;
      }
    }
  }
  if (!rt::device_available(opts.device)) {
    fprintf(stderr, "device %s not available\n", rt::device_name(opts.device));
    return 2;
  }
  setvbuf(stdout, nullptr, _IOLBF, 0);   // live progress when piped (ssh)
  printf("device: %s  seq: %d  requests/user: %d\n",
         rt::device_name(opts.device), seq, requests);

  rt::Model model = rt::Model::load(argv[1]);
  rt::forward(model, rt_bench::synth(1, seq, 1), opts);  // warm (ctx + buffers)

  printf("\n%5s %10s %10s %9s %9s %9s %9s\n", "users", "req/s", "tok/s",
         "p50 ms", "p95 ms", "max ms", "speedup");
  double base_rps = 0;
  for (int nu : users) {
    // Pre-generate batches outside the timed region: synth() is benchmark
    // scaffolding, not serving work (a real server receives the context).
    std::vector<std::vector<rt::Batch>> work(nu);
    {
      std::vector<std::thread> gen;
      for (int u = 0; u < nu; u++)
        gen.emplace_back([&, u] {
          work[u].reserve(requests);
          for (int r = 0; r < requests; r++)
            work[u].push_back(rt_bench::synth(1, seq, 100u + u * 1000u + r));
        });
      for (auto& t : gen) t.join();
    }
    std::vector<std::vector<double>> lat(nu);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (int u = 0; u < nu; u++)
      ts.emplace_back([&, u] {
        lat[u].reserve(requests);
        for (int r = 0; r < requests; r++) {
          auto r0 = std::chrono::steady_clock::now();
          rt::Output o = rt::forward(model, work[u][r], opts);
          lat[u].push_back(ms_since(r0));
          if (o.yhat_number.empty()) { fprintf(stderr, "empty output\n"); abort(); }
        }
      });
    for (auto& t : ts) t.join();
    double wall_ms = ms_since(t0);
    std::vector<double> all;
    for (auto& l : lat) all.insert(all.end(), l.begin(), l.end());
    double rps = all.size() / (wall_ms / 1e3);
    double tok = rps * seq;
    if (nu == users.front()) base_rps = rps;
    printf("%5d %10.2f %10.0f %9.1f %9.1f %9.1f %8.2fx\n", nu, rps, tok,
           pct(all, 0.50), pct(all, 0.95), pct(all, 1.0), rps / base_rps);
  }
  return 0;
}
