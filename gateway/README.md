# RelativeDB gateway

This is the public control and billing layer in front of `rt_serve`. It runs
as a persistent Fly Machine, discovers Vast (or any other) workers through
AWS Cloud Map, authenticates API keys, and atomically records per-customer
usage in DynamoDB.

## Architecture

`api.relativedb.com -> Fly TLS/proxy -> gateway -> Cloud Map -> rt_serve workers`

API keys are never stored in plaintext. The keys table contains SHA-256
digests and customer IDs. The usage table contains an expiring request ledger
and non-expiring monthly customer totals. Each completed request meters:

- `data_in_bytes`: exact HTTP request body length
- `data_out_bytes`: exact upstream response body length
- `model_tokens`: `b * s` for `/v1/forward` and `/v2/forward`
- request count, status, path, and gateway latency

The backend is a scorer, not a text generator, so it has no honest
"completion token" unit. `model_tokens` measures transformer work; the two
byte counters support ingress/egress pricing.

## Deploy AWS backing services

```sh
aws cloudformation deploy --template-file gateway/infra-fly.yaml \
  --stack-name relativedb-gateway-core --capabilities CAPABILITY_NAMED_IAM \
  --profile personal --region us-west-2
```

The stack creates the Cloud Map registry, DynamoDB tables, and a least-
privilege IAM role trusted only by the `relativedb-gateway` Fly app through
short-lived OIDC credentials. No permanent AWS key is stored at Fly.

## Deploy Fly

```sh
cd gateway
flyctl deploy --remote-only --ha=false
flyctl certs add api.relativedb.com
```

The checked-in configuration keeps one 512 MB `shared-cpu-1x` Machine running
in San Jose. Fly terminates and renews TLS. Route 53 has A/AAAA records for
Fly's shared Anycast addresses, so there is no dedicated IPv4 charge.

## Manage access and workers

Create a customer key (the plaintext is printed once; local boto3 required):

```sh
python gateway/manage.py --profile personal --region us-west-2 create-key \
  --table relativedb-gateway-api-keys --customer CUSTOMER_ID
```

Register each Vast worker using a stable instance ID and its externally
reachable `rt_serve` URL:

```sh
python gateway/manage.py --profile personal --region us-west-2 register-worker \
  --service-id srv-afwgq7ghit7s35ko --instance-id vast-123 \
  --url https://worker.example:8500
```

Cloud Map custom health is initially healthy. Deregister a worker before
stopping it. The gateway also retries connection failures against up to three
discovered workers.

Clients pass either `Authorization: Bearer rdb_...` or `X-Api-Key: rdb_...`.
For example, `RemoteBackend` can use:

```python
RemoteBackend(
    "https://api.relativedb.com",
    schema=schema,
    headers={"Authorization": f"Bearer {os.environ['RELATIVEDB_API_KEY']}"},
)
```

The gateway permits a five-minute upstream request and uses connection-based
concurrency limits, which accommodates long inference polls without
duration-based Lambda charges. Cloud Map results are cached for five seconds.

Only register HTTPS worker URLs unless the worker is reached through a private
network. A public HTTP worker would expose request data between Fly and Vast.

`template.yaml` remains as an optional Lambda/API Gateway deployment for
short-running workloads.
