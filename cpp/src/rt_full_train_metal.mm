// rt_full_train_metal.mm -- end-to-end RT-J optimization on Apple Metal.
//
// Dense forward/backward/weight-gradient products use
// MPSMatrixMultiplication.  Relational attention stays sparse and is
// differentiated by custom Metal kernels over the exact query/key groups
// produced by detail::prepare; no SxS mask is materialized.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "rt_internal.hpp"
#include "rt_train.hpp"

namespace rt {
namespace {

constexpr int D = kDModel;
constexpr int H = kHeads;
constexpr int HD = kHeadDim;

const char* kFullTrainMsl = R"MSL(
#include <metal_stdlib>
using namespace metal;
constant float EPS = 1e-6f;

struct Work { int qs, nq, ks, nk, base; float logkv; };
struct AdamArgs { uint n, step; float lr, wd, b1, b2, eps, clip; };

kernel void fill_zero(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) { if (i<n) x[i]=0; }
kernel void copy_f32(device const float* a [[buffer(0)]], device float* b [[buffer(1)]],
                     constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i<n) b[i]=a[i];
}
kernel void add_f32(device const float* a [[buffer(0)]], device float* b [[buffer(1)]],
                    constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i<n) b[i]+=a[i];
}

kernel void rms_fwd(device const float* x [[buffer(0)]], device const float* s [[buffer(1)]],
                    device float* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  x += (ulong)row*n; y += (ulong)row*n;
  float ss=0; for(uint i=lane;i<n;i+=32) ss+=x[i]*x[i];
  float inv=rsqrt(simd_sum(ss)/float(n)+EPS);
  for(uint i=lane;i<n;i+=32) y[i]=x[i]*inv*s[i];
}
kernel void rms_dx(device const float* x [[buffer(0)]], device const float* s [[buffer(1)]],
                   device const float* dy [[buffer(2)]], device float* dx [[buffer(3)]],
                   constant uint& n [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]]) {
  x+=(ulong)row*n; dy+=(ulong)row*n; dx+=(ulong)row*n;
  float ss=0, dot=0;
  for(uint i=lane;i<n;i+=32){ ss+=x[i]*x[i]; dot+=dy[i]*s[i]*x[i]; }
  float inv=rsqrt(simd_sum(ss)/float(n)+EPS);
  float c=simd_sum(dot)*inv*inv/float(n);
  for(uint i=lane;i<n;i+=32) dx[i]+=inv*(dy[i]*s[i]-x[i]*c);
}
kernel void rms_ds(device const float* y [[buffer(0)]], device const float* s [[buffer(1)]],
                   device const float* dy [[buffer(2)]], device float* ds [[buffer(3)]],
                   constant uint2& shape [[buffer(4)]],
                   uint d [[thread_position_in_grid]]) {
  if(d>=shape.y) return; float z=0;
  float invs=1.0f/s[d];
  for(uint r=0;r<shape.x;r++) z+=dy[(ulong)r*shape.y+d]*y[(ulong)r*shape.y+d]*invs;
  ds[d]+=z;
}

kernel void qnorm_fwd(device const float* x [[buffer(0)]], device const float* s [[buffer(1)]],
                      device float* y [[buffer(2)]], uint seg [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
  device const float* a=x+(ulong)seg*64; device float* b=y+(ulong)seg*64;
  float u=a[lane],v=a[lane+32]; float inv=rsqrt(simd_sum(u*u+v*v)/64.0f+EPS);
  b[lane]=u*inv*s[lane]; b[lane+32]=v*inv*s[lane+32];
}
kernel void qnorm_dx(device const float* x [[buffer(0)]], device const float* s [[buffer(1)]],
                     device const float* dy [[buffer(2)]], device float* dx [[buffer(3)]],
                     uint seg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  x+=(ulong)seg*64; dy+=(ulong)seg*64; dx+=(ulong)seg*64;
  float u=x[lane],v=x[lane+32];
  float inv=rsqrt(simd_sum(u*u+v*v)/64.0f+EPS);
  float dot=simd_sum(dy[lane]*s[lane]*u+dy[lane+32]*s[lane+32]*v);
  float c=dot*inv*inv/64.0f;
  dx[lane]+=inv*(dy[lane]*s[lane]-u*c);
  dx[lane+32]+=inv*(dy[lane+32]*s[lane+32]-v*c);
}
kernel void qnorm_ds(device const float* y [[buffer(0)]], device const float* s [[buffer(1)]],
                     device const float* dy [[buffer(2)]], device float* ds [[buffer(3)]],
                     constant uint& nseg [[buffer(4)]],
                     uint d [[thread_position_in_grid]]) {
  if(d>=64) return; float z=0,invs=1.0f/s[d];
  for(uint seg=0;seg<nseg;seg++) z+=dy[(ulong)seg*64+d]*y[(ulong)seg*64+d]*invs;
  ds[d]+=z;
}

kernel void attn_fwd(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                     device const float* v [[buffer(2)]], device float* out [[buffer(3)]],
                     device const int* qi [[buffer(4)]], device const int* ki [[buffer(5)]],
                     device const Work* works [[buffer(6)]], device const float* hs [[buffer(7)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  Work w=works[tg];
  for(uint p=sg;p<uint(w.nq)*8;p+=nsg){
    uint r=p/8,h=p%8,qr=w.base+qi[w.qs+r]; float sc=hs[h]*w.logkv/64.0f;
    device const float* qq=q+(ulong)qr*512+h*64;
    float q0=qq[2*lane]*sc,q1=qq[2*lane+1]*sc,m=-INFINITY,z=0,o0=0,o1=0;
    for(int j=0;j<w.nk;j++){
      uint kr=w.base+ki[w.ks+j]; device const float* kk=k+(ulong)kr*512+h*64;
      float score=simd_sum(q0*kk[2*lane]+q1*kk[2*lane+1]);
      float nm=max(m,score),a=exp(m-nm),b=exp(score-nm); z=z*a+b;
      device const float* vv=v+(ulong)kr*512+h*64;
      o0=o0*a+b*vv[2*lane]; o1=o1*a+b*vv[2*lane+1]; m=nm;
    }
    device float* oo=out+(ulong)qr*512+h*64; oo[2*lane]=o0/z; oo[2*lane+1]=o1/z;
  }
}

kernel void attn_bwd(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                     device const float* v [[buffer(2)]], device const float* out [[buffer(3)]],
                     device const float* dout [[buffer(4)]], device float* dq [[buffer(5)]],
                     device atomic_float* dk [[buffer(6)]], device atomic_float* dv [[buffer(7)]],
                     device const int* qi [[buffer(8)]], device const int* ki [[buffer(9)]],
                     device const Work* works [[buffer(10)]], device const float* hs [[buffer(11)]],
                     device atomic_float* dhs [[buffer(12)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  Work w=works[tg];
  for(uint p=sg;p<uint(w.nq)*8;p+=nsg){
    uint r=p/8,h=p%8,qr=w.base+qi[w.qs+r]; float sc=hs[h]*w.logkv/64.0f;
    device const float* qq=q+(ulong)qr*512+h*64;
    device const float* go=dout+(ulong)qr*512+h*64;
    device const float* oo=out+(ulong)qr*512+h*64;
    float q0=qq[2*lane],q1=qq[2*lane+1],mx=-INFINITY,z=0;
    for(int j=0;j<w.nk;j++){
      uint kr=w.base+ki[w.ks+j]; device const float* kk=k+(ulong)kr*512+h*64;
      mx=max(mx,simd_sum(q0*sc*kk[2*lane]+q1*sc*kk[2*lane+1]));
    }
    for(int j=0;j<w.nk;j++){
      uint kr=w.base+ki[w.ks+j]; device const float* kk=k+(ulong)kr*512+h*64;
      z+=exp(simd_sum(q0*sc*kk[2*lane]+q1*sc*kk[2*lane+1])-mx);
    }
    float gq0=0,gq1=0,gh=0;
    for(int j=0;j<w.nk;j++){
      uint kr=w.base+ki[w.ks+j]; device const float* kk=k+(ulong)kr*512+h*64;
      device const float* vv=v+(ulong)kr*512+h*64;
      float dotq=simd_sum(q0*kk[2*lane]+q1*kk[2*lane+1]);
      float prob=exp(dotq*sc-mx)/z;
      float ds=prob*simd_sum(go[2*lane]*(vv[2*lane]-oo[2*lane])+
                             go[2*lane+1]*(vv[2*lane+1]-oo[2*lane+1]));
      gq0+=ds*sc*kk[2*lane]; gq1+=ds*sc*kk[2*lane+1];
      atomic_fetch_add_explicit(dk+(ulong)kr*512+h*64+2*lane,ds*sc*q0,memory_order_relaxed);
      atomic_fetch_add_explicit(dk+(ulong)kr*512+h*64+2*lane+1,ds*sc*q1,memory_order_relaxed);
      atomic_fetch_add_explicit(dv+(ulong)kr*512+h*64+2*lane,prob*go[2*lane],memory_order_relaxed);
      atomic_fetch_add_explicit(dv+(ulong)kr*512+h*64+2*lane+1,prob*go[2*lane+1],memory_order_relaxed);
      gh+=ds*dotq*w.logkv/64.0f;
    }
    device float* qg=dq+(ulong)qr*512+h*64; qg[2*lane]=gq0; qg[2*lane+1]=gq1;
    if(lane==0) atomic_fetch_add_explicit(dhs+h,gh,memory_order_relaxed);
  }
}

kernel void gate_fwd(device const float* a [[buffer(0)]], device const float* g [[buffer(1)]],
                     device float* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
  if(i<n) y[i]=a[i]*(2.0f/(1.0f+exp(-g[i])));
}
kernel void gate_bwd(device const float* a [[buffer(0)]], device const float* g [[buffer(1)]],
                     device const float* dy [[buffer(2)]], device float* da [[buffer(3)]],
                     device float* dg [[buffer(4)]], constant uint& n [[buffer(5)]],
                     uint i [[thread_position_in_grid]]) {
  if(i<n){float s=1.0f/(1.0f+exp(-g[i])); da[i]=dy[i]*2*s; dg[i]=dy[i]*a[i]*2*s*(1-s);}
}
kernel void swiglu_fwd(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                       device float* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
  if(i<n){float s=1.0f/(1.0f+exp(-a[i])); y[i]=a[i]*s*b[i];}
}
kernel void swiglu_bwd(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                       device const float* dy [[buffer(2)]], device float* da [[buffer(3)]],
                       device float* db [[buffer(4)]], constant uint& n [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
  if(i<n){float s=1.0f/(1.0f+exp(-a[i])); da[i]=dy[i]*b[i]*s*(1+a[i]*(1-s)); db[i]=dy[i]*a[i]*s;}
}

kernel void huber_targets(device const float* pred [[buffer(0)]], device const float* truth [[buffer(1)]],
                          device const uchar* target [[buffer(2)]], device float* dp [[buffer(3)]],
                          device atomic_float* loss [[buffer(4)]], constant uint2& shape [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
  uint n=shape.x*shape.y; if(i>=n) return; dp[i]=0; if(!target[i]) return;
  float e=pred[i]-truth[i],ae=abs(e),l=ae<1?0.5f*e*e:ae-0.5f;
  float inv=1.0f/float(shape.x); dp[i]=(ae<1?e:copysign(1.0f,e))*inv;
  atomic_fetch_add_explicit(loss,l*inv,memory_order_relaxed);
}

kernel void bias_grad(device const float* dy [[buffer(0)]], device float* db [[buffer(1)]],
                      constant uint2& shape [[buffer(2)]], uint c [[thread_position_in_grid]]) {
  if(c>=shape.y)return; float z=0; for(uint r=0;r<shape.x;r++) z+=dy[(ulong)r*shape.y+c]; db[c]+=z;
}
kernel void add_bias(device float* y [[buffer(0)]], device const float* b [[buffer(1)]],
                     constant uint2& shape [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if(i<shape.x*shape.y)y[i]+=b[i%shape.y];
}
kernel void mask_rows(device const float* x [[buffer(0)]], device const uchar* mask [[buffer(1)]],
                      device float* y [[buffer(2)]], constant uint2& shape [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
  if(i<shape.x*shape.y)y[i]=mask[i/shape.y]?x[i]:0.0f;
}
kernel void mask_embedding_grad(device const float* dx [[buffer(0)]],
                                device const uchar* target_type [[buffer(1)]],
                                device float* grad [[buffer(2)]], constant uint2& shape [[buffer(3)]],
                                uint d [[thread_position_in_grid]]) {
  if(d>=shape.y)return;float z=0;for(uint r=0;r<shape.x;r++)if(target_type[r])z+=dx[(ulong)r*shape.y+d];grad[d]+=z;
}
kernel void grad_square(device const float* g [[buffer(0)]], device atomic_float* sum [[buffer(1)]],
                        constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if(i<n) atomic_fetch_add_explicit(sum,g[i]*g[i],memory_order_relaxed);
}
kernel void adamw(device float* w [[buffer(0)]], device const float* g [[buffer(1)]],
                  device float* m [[buffer(2)]], device float* v [[buffer(3)]],
                  constant AdamArgs& a [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if(i>=a.n)return; float gg=g[i]*a.clip;
  float mm=a.b1*m[i]+(1-a.b1)*gg, vv=a.b2*v[i]+(1-a.b2)*gg*gg; m[i]=mm;v[i]=vv;
  float mh=mm/(1-pow(a.b1,float(a.step))),vh=vv/(1-pow(a.b2,float(a.step)));
  w[i]-=a.lr*(mh/(sqrt(vh)+a.eps)+a.wd*w[i]);
}
)MSL";

struct WorkGpu { int qs, nq, ks, nk, base; float logkv; };
struct AdamHost { uint32_t n, step; float lr, wd, b1, b2, eps, clip; };

struct Param {
  std::string key;
  id<MTLBuffer> w, g, m, v;
  size_t n = 0;
  bool used = false;
};

struct FullCtx {
  std::mutex mu;
  id<MTLDevice> dev;
  id<MTLCommandQueue> queue;
  id<MTLCommandBuffer> active;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipe;
  std::unordered_map<std::string, Param> params;
  uint64_t step = 0;
  uint32_t accumulated_microbatches = 0;
};

id<MTLBuffer> buffer(id<MTLDevice> d, size_t n) {
  return [d newBufferWithLength:std::max<size_t>(n, 4)
                         options:MTLResourceStorageModeShared];
}
id<MTLBuffer> upload(id<MTLDevice> d, const void* p, size_t n) {
  return [d newBufferWithBytes:p length:std::max<size_t>(n, 4)
                        options:MTLResourceStorageModeShared];
}

FullCtx* make_full_ctx(Model& model) {
  auto* c = new FullCtx;
  c->dev = MTLCreateSystemDefaultDevice();
  if (!c->dev) throw std::runtime_error("rt/full-train: no Metal device");
  c->queue = [c->dev newCommandQueue];
  NSError* err=nil; MTLCompileOptions* options=[MTLCompileOptions new];
  if (@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
  id<MTLLibrary> lib=[c->dev newLibraryWithSource:@(kFullTrainMsl) options:options error:&err];
  if(!lib) throw std::runtime_error(std::string("rt/full-train: shader compile: ")+
      (err?err.localizedDescription.UTF8String:"?"));
  const char* names[]={"fill_zero","copy_f32","add_f32","rms_fwd","rms_dx","rms_ds",
    "qnorm_fwd","qnorm_dx","qnorm_ds","attn_fwd","attn_bwd","gate_fwd","gate_bwd",
    "swiglu_fwd","swiglu_bwd","huber_targets","bias_grad","add_bias","mask_rows",
    "mask_embedding_grad","grad_square","adamw"};
  for(const char* s:names){
    NSString* ns=[NSString stringWithUTF8String:s]; id<MTLFunction> fn=[lib newFunctionWithName:ns];
    id<MTLComputePipelineState> ps=[c->dev newComputePipelineStateWithFunction:fn error:&err];
    if(!ps) throw std::runtime_error(std::string("rt/full-train: pipeline ")+s);
    c->pipe.emplace(s,ps);
  }
  for(auto& [key,t]:model.store){
    if(t.qtype!=(uint8_t)WType::F32)
      throw std::runtime_error("full-model fine-tuning requires an unquantized checkpoint: "+key);
    Param p; p.key=key;p.n=t.data.size();p.w=upload(c->dev,t.data.data(),p.n*4);
    p.g=buffer(c->dev,p.n*4);p.m=buffer(c->dev,p.n*4);p.v=buffer(c->dev,p.n*4);
    std::memset(p.m.contents,0,p.n*4);std::memset(p.v.contents,0,p.n*4);
    c->params.emplace(key,std::move(p));
  }
  return c;
}

Param& P(FullCtx& c,const std::string& key){
  auto it=c.params.find(key);if(it==c.params.end())throw std::runtime_error("missing train parameter "+key);
  it->second.used=true;return it->second;
}

void wait_ok(id<MTLCommandBuffer> cb) {
  [cb commit];[cb waitUntilCompleted];
  if(cb.status==MTLCommandBufferStatusError)
    throw std::runtime_error(std::string("rt/full-train: Metal command failed: ")+
      (cb.error?cb.error.localizedDescription.UTF8String:"?"));
}

id<MTLCommandBuffer> command(FullCtx& c) {
  if (!c.active) c.active=[c.queue commandBuffer];
  return c.active;
}
void flush(FullCtx& c) {
  if (!c.active) return;
  id<MTLCommandBuffer> cb=c.active;c.active=nil;wait_ok(cb);
}

template<class Bind>
void kernel(FullCtx& c,const char* name,size_t threads,Bind bind,bool simd=false){
  if(!threads)return; id<MTLCommandBuffer> cb=command(c);
  id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
  [e setComputePipelineState:c.pipe.at(name)];bind(e);
  if(simd)[e dispatchThreadgroups:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
  else [e dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<size_t>(256,threads),1,1)];
  [e endEncoding];
}

void zero(FullCtx& c,id<MTLBuffer> x,size_t n){uint32_t nn=(uint32_t)n;kernel(c,"fill_zero",n,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:x offset:0 atIndex:0];[e setBytes:&nn length:4 atIndex:1];});}
void copy(FullCtx& c,id<MTLBuffer>a,id<MTLBuffer>b,size_t n){uint32_t nn=n;kernel(c,"copy_f32",n,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:a offset:0 atIndex:0];[e setBuffer:b offset:0 atIndex:1];[e setBytes:&nn length:4 atIndex:2];});}
void add(FullCtx& c,id<MTLBuffer>a,id<MTLBuffer>b,size_t n){uint32_t nn=n;kernel(c,"add_f32",n,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:a offset:0 atIndex:0];[e setBuffer:b offset:0 atIndex:1];[e setBytes:&nn length:4 atIndex:2];});}

void gemm(FullCtx& c,id<MTLBuffer>a,id<MTLBuffer>b,id<MTLBuffer>o,
          int M,int N,int K,bool ta,bool tb,float beta=0){
  id<MTLCommandBuffer> cb=command(c);
  MPSMatrixMultiplication* mm=[[MPSMatrixMultiplication alloc]initWithDevice:c.dev
    transposeLeft:ta transposeRight:tb resultRows:M resultColumns:N interiorColumns:K alpha:1 beta:beta];
  auto desc=[](int r,int col){return [MPSMatrixDescriptor matrixDescriptorWithRows:r columns:col rowBytes:(size_t)col*4 dataType:MPSDataTypeFloat32];};
  // Descriptor shapes describe stored matrices, not their logical transposes.
  MPSMatrix* A=[[MPSMatrix alloc]initWithBuffer:a descriptor:desc(ta?K:M,ta?M:K)];
  MPSMatrix* B=[[MPSMatrix alloc]initWithBuffer:b descriptor:desc(tb?N:K,tb?K:N)];
  MPSMatrix* O=[[MPSMatrix alloc]initWithBuffer:o descriptor:desc(M,N)];
  [mm encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B resultMatrix:O];
}

struct GroupBuffers { id<MTLBuffer> qi,ki,w; size_t nw=0; };
GroupBuffers groups(FullCtx& c,const detail::Prepared& prep,int which){
  const std::vector<detail::Groups>* all[3]={&prep.g_col,&prep.g_feat,&prep.g_nbr};
  std::vector<int32_t> q,k;std::vector<WorkGpu>w;
  for(int b=0;b<prep.B;b++){
    const auto&G=(*all[which])[b];int qb=q.size(),kb=k.size();
    q.insert(q.end(),G.q.begin(),G.q.end());k.insert(k.end(),G.k.begin(),G.k.end());
    for(int g=0;g<G.n();g++){
      int nq=G.qoff[g+1]-G.qoff[g],nk=G.koff[g+1]-G.koff[g];
      for(int q0=0;q0<nq;q0+=detail::kQTile)
        w.push_back({qb+G.qoff[g]+q0,std::min(detail::kQTile,nq-q0),kb+G.koff[g],nk,b*prep.S,
                     std::log(std::max(1.f,detail::bf16_round((float)nk)))});
    }
  }
  return {upload(c.dev,q.data(),q.size()*4),upload(c.dev,k.data(),k.size()*4),
          upload(c.dev,w.data(),w.size()*sizeof(WorkGpu)),w.size()};
}

struct AttnTape { id<MTLBuffer> x,xn,q,k,v,g,qn,kn,a,y,out; };
struct FfnTape { id<MTLBuffer> x,xn,a,b,y,out; };

void rms_forward(FullCtx&c,id<MTLBuffer>x,Param&s,id<MTLBuffer>y,size_t rows,uint32_t n){
  kernel(c,"rms_fwd",rows,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:x offset:0 atIndex:0];[e setBuffer:s.w offset:0 atIndex:1];[e setBuffer:y offset:0 atIndex:2];[e setBytes:&n length:4 atIndex:3];},true);
}
void rms_backward(FullCtx&c,id<MTLBuffer>x,id<MTLBuffer>y,Param&s,id<MTLBuffer>dy,id<MTLBuffer>dx,size_t rows,uint32_t n){
  kernel(c,"rms_dx",rows,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:x offset:0 atIndex:0];[e setBuffer:s.w offset:0 atIndex:1];[e setBuffer:dy offset:0 atIndex:2];[e setBuffer:dx offset:0 atIndex:3];[e setBytes:&n length:4 atIndex:4];},true);
  uint32_t shape[2]={(uint32_t)rows,n};kernel(c,"rms_ds",n,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:y offset:0 atIndex:0];[e setBuffer:s.w offset:0 atIndex:1];[e setBuffer:dy offset:0 atIndex:2];[e setBuffer:s.g offset:0 atIndex:3];[e setBytes:shape length:8 atIndex:4];});
}

std::string ap(int b,int a){static const char*n[]={"col","feat","nbr"};return "blocks."+std::to_string(b)+".attns."+n[a]+".";}
std::string np(int b,int a){static const char*n[]={"col","feat","nbr","ffn"};return "blocks."+std::to_string(b)+".norms."+n[a]+".scale";}

AttnTape attention_forward(FullCtx&c,int b,int a,id<MTLBuffer>x,size_t rows,const GroupBuffers&gb){
  const size_t n=rows*D,bytes=n*4;AttnTape t;t.x=x;t.xn=buffer(c.dev,bytes);
  rms_forward(c,x,P(c,np(b,a)),t.xn,rows,D);t.q=buffer(c.dev,bytes);t.k=buffer(c.dev,bytes);
  t.v=buffer(c.dev,bytes);t.g=buffer(c.dev,bytes);std::string p=ap(b,a);
  gemm(c,t.xn,P(c,p+"wq.weight").w,t.q,rows,D,D,false,true);
  gemm(c,t.xn,P(c,p+"wk.weight").w,t.k,rows,D,D,false,true);
  gemm(c,t.xn,P(c,p+"wv.weight").w,t.v,rows,D,D,false,true);
  gemm(c,t.xn,P(c,p+"wg.weight").w,t.g,rows,D,D,false,true);
  t.qn=buffer(c.dev,bytes);t.kn=buffer(c.dev,bytes);Param&qns=P(c,p+"q_norm.scale"),&kns=P(c,p+"k_norm.scale");
  kernel(c,"qnorm_fwd",rows*H,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:t.q offset:0 atIndex:0];[e setBuffer:qns.w offset:0 atIndex:1];[e setBuffer:t.qn offset:0 atIndex:2];},true);
  kernel(c,"qnorm_fwd",rows*H,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:t.k offset:0 atIndex:0];[e setBuffer:kns.w offset:0 atIndex:1];[e setBuffer:t.kn offset:0 atIndex:2];},true);
  t.a=buffer(c.dev,bytes);zero(c,t.a,n);Param&hs=P(c,p+"scale");
  kernel(c,"attn_fwd",gb.nw,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:t.qn offset:0 atIndex:0];[e setBuffer:t.kn offset:0 atIndex:1];[e setBuffer:t.v offset:0 atIndex:2];[e setBuffer:t.a offset:0 atIndex:3];[e setBuffer:gb.qi offset:0 atIndex:4];[e setBuffer:gb.ki offset:0 atIndex:5];[e setBuffer:gb.w offset:0 atIndex:6];[e setBuffer:hs.w offset:0 atIndex:7];},true);
  t.y=buffer(c.dev,bytes);uint32_t nn=n;kernel(c,"gate_fwd",n,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:t.a offset:0 atIndex:0];[e setBuffer:t.g offset:0 atIndex:1];[e setBuffer:t.y offset:0 atIndex:2];[e setBytes:&nn length:4 atIndex:3];});
  t.out=buffer(c.dev,bytes);copy(c,x,t.out,n);gemm(c,t.y,P(c,p+"wo.weight").w,t.out,rows,D,D,false,true,1);return t;
}

FfnTape ffn_forward(FullCtx&c,int b,id<MTLBuffer>x,size_t rows){
  const size_t nd=rows*D,nf=rows*kDFF;FfnTape t;t.x=x;t.xn=buffer(c.dev,nd*4);
  rms_forward(c,x,P(c,np(b,3)),t.xn,rows,D);std::string p="blocks."+std::to_string(b)+".ffn.";
  t.a=buffer(c.dev,nf*4);t.b=buffer(c.dev,nf*4);t.y=buffer(c.dev,nf*4);
  gemm(c,t.xn,P(c,p+"w1.weight").w,t.a,rows,kDFF,D,false,true);
  gemm(c,t.xn,P(c,p+"w3.weight").w,t.b,rows,kDFF,D,false,true);
  uint32_t nn=nf;kernel(c,"swiglu_fwd",nf,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:t.a offset:0 atIndex:0];[e setBuffer:t.b offset:0 atIndex:1];[e setBuffer:t.y offset:0 atIndex:2];[e setBytes:&nn length:4 atIndex:3];});
  t.out=buffer(c.dev,nd*4);copy(c,x,t.out,nd);gemm(c,t.y,P(c,p+"w2.weight").w,t.out,rows,D,kDFF,false,true,1);return t;
}

id<MTLBuffer> block_forward(FullCtx&c,int b,id<MTLBuffer>x,size_t rows,const GroupBuffers gb[3]){
  for(int a=0;a<3;a++)x=attention_forward(c,b,a,x,rows,gb[a]).out;
  return ffn_forward(c,b,x,rows).out;
}

void linear_backward(FullCtx&c,id<MTLBuffer>x,Param&w,id<MTLBuffer>dy,id<MTLBuffer>dx,int M,int N,int K){
  gemm(c,dy,w.w,dx,M,K,N,false,false,1);gemm(c,dy,x,w.g,N,K,M,true,false,1);
}

id<MTLBuffer> ffn_backward(FullCtx&c,int b,const FfnTape&t,id<MTLBuffer>dout,size_t rows){
  size_t nd=rows*D,nf=rows*kDFF;std::string p="blocks."+std::to_string(b)+".ffn.";
  id<MTLBuffer> dx=buffer(c.dev,nd*4);copy(c,dout,dx,nd);id<MTLBuffer>dy=buffer(c.dev,nf*4);zero(c,dy,nf);
  linear_backward(c,t.y,P(c,p+"w2.weight"),dout,dy,rows,D,kDFF);
  id<MTLBuffer>da=buffer(c.dev,nf*4),db=buffer(c.dev,nf*4);uint32_t nn=nf;
  kernel(c,"swiglu_bwd",nf,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:t.a offset:0 atIndex:0];[e setBuffer:t.b offset:0 atIndex:1];[e setBuffer:dy offset:0 atIndex:2];[e setBuffer:da offset:0 atIndex:3];[e setBuffer:db offset:0 atIndex:4];[e setBytes:&nn length:4 atIndex:5];});
  id<MTLBuffer>dxn=buffer(c.dev,nd*4);zero(c,dxn,nd);
  linear_backward(c,t.xn,P(c,p+"w1.weight"),da,dxn,rows,kDFF,D);
  linear_backward(c,t.xn,P(c,p+"w3.weight"),db,dxn,rows,kDFF,D);
  rms_backward(c,t.x,t.xn,P(c,np(b,3)),dxn,dx,rows,D);return dx;
}

id<MTLBuffer> attention_backward(FullCtx&c,int b,int a,const AttnTape&t,id<MTLBuffer>dout,
                                 size_t rows,const GroupBuffers&gb){
  size_t n=rows*D,bytes=n*4;std::string p=ap(b,a);id<MTLBuffer>dx=buffer(c.dev,bytes);copy(c,dout,dx,n);
  id<MTLBuffer>dy=buffer(c.dev,bytes);zero(c,dy,n);linear_backward(c,t.y,P(c,p+"wo.weight"),dout,dy,rows,D,D);
  id<MTLBuffer>da=buffer(c.dev,bytes),dg=buffer(c.dev,bytes);uint32_t nn=n;
  kernel(c,"gate_bwd",n,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:t.a offset:0 atIndex:0];[e setBuffer:t.g offset:0 atIndex:1];[e setBuffer:dy offset:0 atIndex:2];[e setBuffer:da offset:0 atIndex:3];[e setBuffer:dg offset:0 atIndex:4];[e setBytes:&nn length:4 atIndex:5];});
  id<MTLBuffer>dqn=buffer(c.dev,bytes),dkn=buffer(c.dev,bytes),dv=buffer(c.dev,bytes);zero(c,dqn,n);zero(c,dkn,n);zero(c,dv,n);
  Param&hs=P(c,p+"scale");
  kernel(c,"attn_bwd",gb.nw,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:t.qn offset:0 atIndex:0];[e setBuffer:t.kn offset:0 atIndex:1];[e setBuffer:t.v offset:0 atIndex:2];[e setBuffer:t.a offset:0 atIndex:3];[e setBuffer:da offset:0 atIndex:4];[e setBuffer:dqn offset:0 atIndex:5];[e setBuffer:dkn offset:0 atIndex:6];[e setBuffer:dv offset:0 atIndex:7];[e setBuffer:gb.qi offset:0 atIndex:8];[e setBuffer:gb.ki offset:0 atIndex:9];[e setBuffer:gb.w offset:0 atIndex:10];[e setBuffer:hs.w offset:0 atIndex:11];[e setBuffer:hs.g offset:0 atIndex:12];},true);
  id<MTLBuffer>dq=buffer(c.dev,bytes),dk=buffer(c.dev,bytes);zero(c,dq,n);zero(c,dk,n);
  Param&qns=P(c,p+"q_norm.scale"),&kns=P(c,p+"k_norm.scale");uint32_t nseg=rows*H;
  kernel(c,"qnorm_dx",nseg,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:t.q offset:0 atIndex:0];[e setBuffer:qns.w offset:0 atIndex:1];[e setBuffer:dqn offset:0 atIndex:2];[e setBuffer:dq offset:0 atIndex:3];},true);
  kernel(c,"qnorm_dx",nseg,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:t.k offset:0 atIndex:0];[e setBuffer:kns.w offset:0 atIndex:1];[e setBuffer:dkn offset:0 atIndex:2];[e setBuffer:dk offset:0 atIndex:3];},true);
  kernel(c,"qnorm_ds",HD,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:t.qn offset:0 atIndex:0];[e setBuffer:qns.w offset:0 atIndex:1];[e setBuffer:dqn offset:0 atIndex:2];[e setBuffer:qns.g offset:0 atIndex:3];[e setBytes:&nseg length:4 atIndex:4];});
  kernel(c,"qnorm_ds",HD,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:t.kn offset:0 atIndex:0];[e setBuffer:kns.w offset:0 atIndex:1];[e setBuffer:dkn offset:0 atIndex:2];[e setBuffer:kns.g offset:0 atIndex:3];[e setBytes:&nseg length:4 atIndex:4];});
  id<MTLBuffer>dxn=buffer(c.dev,bytes);zero(c,dxn,n);
  linear_backward(c,t.xn,P(c,p+"wq.weight"),dq,dxn,rows,D,D);
  linear_backward(c,t.xn,P(c,p+"wk.weight"),dk,dxn,rows,D,D);
  linear_backward(c,t.xn,P(c,p+"wv.weight"),dv,dxn,rows,D,D);
  linear_backward(c,t.xn,P(c,p+"wg.weight"),dg,dxn,rows,D,D);
  rms_backward(c,t.x,t.xn,P(c,np(b,a)),dxn,dx,rows,D);return dx;
}

id<MTLBuffer> block_backward(FullCtx&c,int b,id<MTLBuffer>x0,id<MTLBuffer>dout,size_t rows,const GroupBuffers gb[3]){
  AttnTape at[3];id<MTLBuffer>x=x0;for(int a=0;a<3;a++){at[a]=attention_forward(c,b,a,x,rows,gb[a]);x=at[a].out;}
  FfnTape ft=ffn_forward(c,b,x,rows);id<MTLBuffer>dx=ffn_backward(c,b,ft,dout,rows);
  for(int a=2;a>=0;a--)dx=attention_backward(c,b,a,at[a],dx,rows,gb[a]);return dx;
}

void encoder_backward(FullCtx&c,const char* name,id<MTLBuffer>input,int in,
                      id<MTLBuffer>mask,id<MTLBuffer>dx,size_t rows){
  const size_t nd=rows*D;std::string base="enc_dict."+std::string(name);
  Param&w=P(c,base+".weight"),&b=P(c,base+".bias"),&scale=P(c,"norm_dict."+std::string(name)+".scale");
  id<MTLBuffer>raw=buffer(c.dev,nd*4);gemm(c,input,w.w,raw,rows,D,in,false,true);
  uint32_t shape[2]={(uint32_t)rows,D};kernel(c,"add_bias",nd,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:raw offset:0 atIndex:0];[e setBuffer:b.w offset:0 atIndex:1];[e setBytes:shape length:8 atIndex:2];});
  id<MTLBuffer>dy=buffer(c.dev,nd*4);kernel(c,"mask_rows",nd,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:dx offset:0 atIndex:0];[e setBuffer:mask offset:0 atIndex:1];[e setBuffer:dy offset:0 atIndex:2];[e setBytes:shape length:8 atIndex:3];});
  id<MTLBuffer>normed=buffer(c.dev,nd*4);rms_forward(c,raw,scale,normed,rows,D);
  id<MTLBuffer>draw=buffer(c.dev,nd*4);zero(c,draw,nd);rms_backward(c,raw,normed,scale,dy,draw,rows,D);
  // Only parameter gradients are needed at the input boundary.
  gemm(c,draw,input,w.g,D,in,rows,true,false,1);
  uint32_t bshape[2]={(uint32_t)rows,D};kernel(c,"bias_grad",D,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:draw offset:0 atIndex:0];[e setBuffer:b.g offset:0 atIndex:1];[e setBytes:bshape length:8 atIndex:2];});
}

void embedding_backward(FullCtx&c,const Model&model,const Batch&batch,const Output&meta,
                        id<MTLBuffer>dx,size_t rows){
  const size_t BS=rows;std::vector<float>col(BS*kDText),text(BS*kDText);
  std::vector<float>number(BS),datetime(BS),booleanv(BS);std::vector<uint8_t>colmask(BS),masks[4],targets[4];
  for(int t=0;t<4;t++){masks[t].resize(BS);targets[t].resize(BS);}
  for(int b=0;b<batch.B;b++)for(int s=0;s<batch.S;s++){
    size_t dst=(size_t)b*batch.S+s,src=(size_t)b*batch.S+meta.sort_idxs[dst];bool valid=!batch.is_padding[src];
    colmask[dst]=valid;std::memcpy(col.data()+dst*kDText,batch.col_name_v.data()+src*kDText,kDText*4);
    std::memcpy(text.data()+dst*kDText,batch.text_v.data()+src*kDText,kDText*4);
    number[dst]=batch.number_v[src];datetime[dst]=batch.datetime_v[src];booleanv[dst]=batch.boolean_v[src];
    int sem=(int)batch.sem_types[src];if(sem>=0&&sem<4){masks[sem][dst]=valid&&!batch.is_target[src];targets[sem][dst]=valid&&batch.is_target[src];}
  }
  id<MTLBuffer>bcol=upload(c.dev,col.data(),col.size()*4),btext=upload(c.dev,text.data(),text.size()*4);
  id<MTLBuffer>bn=upload(c.dev,number.data(),number.size()*4),bd=upload(c.dev,datetime.data(),datetime.size()*4),bb=upload(c.dev,booleanv.data(),booleanv.size()*4);
  encoder_backward(c,"col_name",bcol,kDText,upload(c.dev,colmask.data(),BS),dx,BS);
  id<MTLBuffer>inputs[4]={bn,btext,bd,bb};int widths[4]={1,kDText,1,1};static const char*names[]={"number","text","datetime","boolean"};
  uint32_t shape[2]={(uint32_t)BS,D};
  for(int t=0;t<4;t++){
    encoder_backward(c,names[t],inputs[t],widths[t],upload(c.dev,masks[t].data(),BS),dx,BS);
    Param&me=P(c,"mask_embs."+std::string(names[t]));id<MTLBuffer>tm=upload(c.dev,targets[t].data(),BS);
    kernel(c,"mask_embedding_grad",D,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:dx offset:0 atIndex:0];[e setBuffer:tm offset:0 atIndex:1];[e setBuffer:me.g offset:0 atIndex:2];[e setBytes:shape length:8 atIndex:3];});
  }
}

float gradient_norm(FullCtx&c,float scale){
  id<MTLBuffer>sum=buffer(c.dev,4);zero(c,sum,1);
  for(auto&[_,p]:c.params)if(p.used){uint32_t n=p.n;kernel(c,"grad_square",p.n,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:p.g offset:0 atIndex:0];[e setBuffer:sum offset:0 atIndex:1];[e setBytes:&n length:4 atIndex:2];});}
  flush(c);
  return std::sqrt(std::max(0.f,*(float*)sum.contents))*scale;
}

float update_parameters(FullCtx&c,const FullFineTuneOptions&o,uint32_t microbatches){
  const float average=1.f/std::max(1u,microbatches);
  float norm=gradient_norm(c,average);
  float clip=average*((o.grad_clip_norm>0&&norm>o.grad_clip_norm)?o.grad_clip_norm/norm:1.f);
  c.step++;
  for(auto&[_,p]:c.params)if(p.used){
    // Match common AdamW practice: matrices decay; biases and norm/scale
    // vectors do not. This also avoids shrinking learned RMS scales.
    bool matrix=false; // recover rank from byte count/key convention below
    matrix=p.key.ends_with(".weight")&&p.n>D;
    AdamHost a{(uint32_t)p.n,(uint32_t)c.step,o.learning_rate,matrix?o.weight_decay:0.f,o.beta1,o.beta2,o.epsilon,clip};
    kernel(c,"adamw",p.n,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:p.w offset:0 atIndex:0];[e setBuffer:p.g offset:0 atIndex:1];[e setBuffer:p.m offset:0 atIndex:2];[e setBuffer:p.v offset:0 atIndex:3];[e setBytes:&a length:sizeof(a) atIndex:4];});
  }
  flush(c);
  return norm;
}

void sync_model(Model&model,FullCtx&c){
  for(auto&[key,p]:c.params)std::memcpy(model.store.at(key).data.data(),p.w.contents,p.n*4);
  // Rebuild all pointer views and fused qkvg storage without re-reading disk.
  // Model::load owns this wiring logic; pointer addresses in store are stable,
  // while fused matrices need their four updated source matrices copied below.
  static const char* an[]={"col","feat","nbr"};
  for(int b=0;b<kBlocks;b++)for(int a=0;a<3;a++){
    std::string p="blocks."+std::to_string(b)+".attns."+an[a]+".";
    for(int j=0;j<4;j++){
      static const char* wn[]={"wq","wk","wv","wg"};
      auto&src=model.store.at(p+wn[j]+".weight").data;
      std::memcpy(model.blocks[b].attn[a].wqkvg_f32.data()+(size_t)j*D*D,src.data(),src.size()*4);
    }
  }
  for(auto&slot:model.device_ctx)slot.reset();
}

float model_loss_locked(FullCtx&c,Model&model,const Batch&batch){
  Output meta;detail::Prepared prep=detail::prepare(model,batch,meta,false);
  size_t rows=(size_t)prep.B*prep.S,n=rows*D;
  GroupBuffers gb[3]={groups(c,prep,0),groups(c,prep,1),groups(c,prep,2)};
  id<MTLBuffer>x=upload(c.dev,prep.x.data(),n*4);
  for(int b=0;b<kBlocks;b++)x=block_forward(c,b,x,rows,gb);
  id<MTLBuffer>xn=buffer(c.dev,n*4);
  rms_forward(c,x,P(c,"norm_out.scale"),xn,rows,D);
  Param&dw=P(c,"dec_dict.number.weight"),&db=P(c,"dec_dict.number.bias");
  id<MTLBuffer>pred=buffer(c.dev,rows*4);
  gemm(c,xn,dw.w,pred,rows,1,D,false,true);
  uint32_t pshape[2]={(uint32_t)rows,1};
  kernel(c,"add_bias",rows,[&](id<MTLComputeCommandEncoder>e){
    [e setBuffer:pred offset:0 atIndex:0];[e setBuffer:db.w offset:0 atIndex:1];
    [e setBytes:pshape length:8 atIndex:2];});
  std::vector<float>truth(rows);
  for(int b=0;b<batch.B;b++)for(int s=0;s<batch.S;s++){
    size_t dst=(size_t)b*batch.S+s,src=(size_t)b*batch.S+meta.sort_idxs[dst];
    truth[dst]=batch.number_v[src];
    if(meta.sorted_is_target[dst]&&batch.sem_types[src]!=kNumber)
      throw std::runtime_error("native full fine-tuning currently requires number/bool-as-number targets");
  }
  id<MTLBuffer>bt=upload(c.dev,truth.data(),rows*4);
  id<MTLBuffer>bm=upload(c.dev,meta.sorted_is_target.data(),rows);
  id<MTLBuffer>dp=buffer(c.dev,rows*4),loss=buffer(c.dev,4);zero(c,loss,1);
  uint32_t shape[2]={(uint32_t)batch.B,(uint32_t)batch.S};
  kernel(c,"huber_targets",rows,[&](id<MTLComputeCommandEncoder>e){
    [e setBuffer:pred offset:0 atIndex:0];[e setBuffer:bt offset:0 atIndex:1];
    [e setBuffer:bm offset:0 atIndex:2];[e setBuffer:dp offset:0 atIndex:3];
    [e setBuffer:loss offset:0 atIndex:4];[e setBytes:shape length:8 atIndex:5];});
  flush(c);
  return *(float*)loss.contents;
}

} // namespace

FullFineTuneStep fit_model_metal_step(Model& model,const Batch& batch,const FullFineTuneOptions& opts){
  @autoreleasepool {
    if(batch.B<=0||batch.S<=0)throw std::runtime_error("full-model fine-tuning batch is empty");
    if(!(opts.learning_rate>0)||opts.weight_decay<0||opts.beta1<0||opts.beta1>=1||opts.beta2<0||opts.beta2>=1||!(opts.epsilon>0))
      throw std::runtime_error("invalid full-model fine-tuning options");
    if(!model.training_ctx)model.training_ctx.reset(make_full_ctx(model),[](void*p){delete (FullCtx*)p;});
    FullCtx&c=*(FullCtx*)model.training_ctx.get();std::lock_guard<std::mutex>lock(c.mu);auto start=std::chrono::steady_clock::now();
    if(c.accumulated_microbatches==0)
      for(auto&[_,p]:c.params){p.used=false;zero(c,p.g,p.n);}
    Output meta;detail::Prepared prep=detail::prepare(model,batch,meta,false);size_t rows=(size_t)prep.B*prep.S,n=rows*D;
    GroupBuffers gb[3]={groups(c,prep,0),groups(c,prep,1),groups(c,prep,2)};
    id<MTLBuffer>x=upload(c.dev,prep.x.data(),n*4);std::vector<id<MTLBuffer>>boundary{kBlocks+1};boundary[0]=x;
    for(int b=0;b<kBlocks;b++)boundary[b+1]=block_forward(c,b,boundary[b],rows,gb);
    id<MTLBuffer>xn=buffer(c.dev,n*4);rms_forward(c,boundary[kBlocks],P(c,"norm_out.scale"),xn,rows,D);
    Param&dw=P(c,"dec_dict.number.weight"),&db=P(c,"dec_dict.number.bias");id<MTLBuffer>pred=buffer(c.dev,rows*4);
    gemm(c,xn,dw.w,pred,rows,1,D,false,true);
    uint32_t pshape[2]={(uint32_t)rows,1};kernel(c,"add_bias",rows,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:pred offset:0 atIndex:0];[e setBuffer:db.w offset:0 atIndex:1];[e setBytes:pshape length:8 atIndex:2];});
    // Sorted numeric labels/target flags.
    std::vector<float>truth(rows);for(int b=0;b<batch.B;b++)for(int s=0;s<batch.S;s++){
      size_t dst=(size_t)b*batch.S+s,src=(size_t)b*batch.S+meta.sort_idxs[dst];truth[dst]=batch.number_v[src];
      if(meta.sorted_is_target[dst]&&batch.sem_types[src]!=kNumber)
        throw std::runtime_error("native full fine-tuning currently requires number/bool-as-number targets");
    }
    id<MTLBuffer>bt=upload(c.dev,truth.data(),rows*4),bm=upload(c.dev,meta.sorted_is_target.data(),rows);
    id<MTLBuffer>dp=buffer(c.dev,rows*4),loss=buffer(c.dev,4);zero(c,loss,1);uint32_t shape[2]={(uint32_t)batch.B,(uint32_t)batch.S};
    kernel(c,"huber_targets",rows,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:pred offset:0 atIndex:0];[e setBuffer:bt offset:0 atIndex:1];[e setBuffer:bm offset:0 atIndex:2];[e setBuffer:dp offset:0 atIndex:3];[e setBuffer:loss offset:0 atIndex:4];[e setBytes:shape length:8 atIndex:5];});
    id<MTLBuffer>dxn=buffer(c.dev,n*4);zero(c,dxn,n);linear_backward(c,xn,dw,dp,dxn,rows,1,D);
    uint32_t bshape[2]={(uint32_t)rows,1};kernel(c,"bias_grad",1,[&](id<MTLComputeCommandEncoder>e){[e setBuffer:dp offset:0 atIndex:0];[e setBuffer:db.g offset:0 atIndex:1];[e setBytes:bshape length:8 atIndex:2];});
    id<MTLBuffer>dx=buffer(c.dev,n*4);zero(c,dx,n);rms_backward(c,boundary[kBlocks],xn,P(c,"norm_out.scale"),dxn,dx,rows,D);
    for(int b=kBlocks-1;b>=0;b--)dx=block_backward(c,b,boundary[b],dx,rows,gb);
    embedding_backward(c,model,batch,meta,dx,rows);
    c.accumulated_microbatches++;
    float grad_norm=gradient_norm(c,1.f/c.accumulated_microbatches);
    bool updated=opts.apply_update;
    uint32_t accumulated=c.accumulated_microbatches;
    if(updated){grad_norm=update_parameters(c,opts,c.accumulated_microbatches);sync_model(model,c);c.accumulated_microbatches=0;}
    float loss_value=*(float*)loss.contents;if(!std::isfinite(loss_value)||!std::isfinite(grad_norm))
      throw std::runtime_error("rt/full-train: non-finite loss or gradient");
    auto stop=std::chrono::steady_clock::now();return {loss_value,grad_norm,c.step,std::chrono::duration<double>(stop-start).count(),accumulated,updated};
  }
}

void reset_model_metal_optimizer(Model& model){model.training_ctx.reset();}

void save_model_metal_optimizer(Model&model,const std::string&path){
  @autoreleasepool {
    if(!model.training_ctx)model.training_ctx.reset(make_full_ctx(model),[](void*p){delete (FullCtx*)p;});
    FullCtx&c=*(FullCtx*)model.training_ctx.get();std::lock_guard<std::mutex>lock(c.mu);flush(c);
    if(c.accumulated_microbatches!=0)throw std::runtime_error("optimizer state can only be saved at an update boundary");
    std::vector<std::string>keys;keys.reserve(c.params.size());for(auto&[k,_]:c.params)keys.push_back(k);std::sort(keys.begin(),keys.end());
    std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("cannot create optimizer state "+path);
    const char magic[8]={'R','T','O','P','T','0','0','2'};out.write(magic,8);
    uint64_t step=c.step,nkeys=keys.size();out.write((char*)&step,8);out.write((char*)&nkeys,8);
    for(const auto&key:keys){Param&p=c.params.at(key);uint32_t len=key.size();uint64_t n=p.n;
      out.write((char*)&len,4);out.write(key.data(),len);out.write((char*)&n,8);
      out.write((char*)p.m.contents,n*4);out.write((char*)p.v.contents,n*4);
    }
    if(!out)throw std::runtime_error("failed writing optimizer state "+path);
  }
}

void load_model_metal_optimizer(Model&model,const std::string&path){
  @autoreleasepool {
    if(!model.training_ctx)model.training_ctx.reset(make_full_ctx(model),[](void*p){delete (FullCtx*)p;});
    FullCtx&c=*(FullCtx*)model.training_ctx.get();std::lock_guard<std::mutex>lock(c.mu);
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open optimizer state "+path);
    char magic[8];in.read(magic,8);if(std::memcmp(magic,"RTOPT002",8)!=0)throw std::runtime_error("bad optimizer state magic");
    uint64_t step=0,nkeys=0;in.read((char*)&step,8);in.read((char*)&nkeys,8);
    if(nkeys!=c.params.size())throw std::runtime_error("optimizer state parameter count mismatch");
    for(uint64_t i=0;i<nkeys;i++){uint32_t len=0;uint64_t n=0;in.read((char*)&len,4);std::string key(len,'\0');in.read(key.data(),len);in.read((char*)&n,8);
      auto it=c.params.find(key);if(it==c.params.end()||it->second.n!=n)throw std::runtime_error("optimizer state tensor mismatch: "+key);
      Param&p=it->second;p.used=false;std::memset(p.g.contents,0,n*4);in.read((char*)p.m.contents,n*4);in.read((char*)p.v.contents,n*4);
    }
    if(!in)throw std::runtime_error("truncated optimizer state "+path);c.step=step;c.accumulated_microbatches=0;
  }
}

FullGradientCheck check_model_metal_gradients(Model&model,const Batch&batch,float epsilon){
  if(!(epsilon>0))throw std::runtime_error("gradient-check epsilon must be positive");
  reset_model_metal_optimizer(model);
  FullFineTuneOptions opts;opts.apply_update=false;
  (void)fit_model_metal_step(model,batch,opts);
  FullCtx&c=*(FullCtx*)model.training_ctx.get();std::lock_guard<std::mutex>lock(c.mu);flush(c);
  const std::vector<std::pair<std::string,size_t>> probes={
    {"enc_dict.number.weight",17},{"mask_embs.number",23},
    {"blocks.0.attns.col.wq.weight",1000},{"blocks.0.attns.col.wv.weight",2000},
    {"blocks.0.attns.col.q_norm.scale",13},{"blocks.0.attns.col.scale",0},
    {"blocks.11.ffn.w2.weight",12345},{"norm_out.scale",31},
    {"dec_dict.number.weight",47},{"dec_dict.number.bias",0}};
  FullGradientCheck out;
  for(const auto&[key,want]:probes){
    auto it=c.params.find(key);if(it==c.params.end()||it->second.n==0)continue;Param&p=it->second;size_t idx=std::min(want,p.n-1);
    float*weights=(float*)p.w.contents;float*grads=(float*)p.g.contents;float original=weights[idx],analytic=grads[idx];
    weights[idx]=original+epsilon;sync_model(model,c);float plus=model_loss_locked(c,model,batch);
    weights[idx]=original-epsilon;sync_model(model,c);float minus=model_loss_locked(c,model,batch);
    weights[idx]=original;sync_model(model,c);float numeric=(plus-minus)/(2*epsilon);
    float ae=std::abs(analytic-numeric);float re=ae/std::max(1e-3f,std::abs(analytic)+std::abs(numeric));
    out.max_absolute_error=std::max(out.max_absolute_error,ae);out.max_relative_error=std::max(out.max_relative_error,re);out.checked++;
  }
  c.accumulated_microbatches=0;for(auto&[_,p]:c.params){p.used=false;std::memset(p.g.contents,0,p.n*4);}sync_model(model,c);
  return out;
}

} // namespace rt
