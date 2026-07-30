# relativedb-engine

The optional in-process native engine for
[relativedb](https://pypi.org/project/relativedb/).

The base `relativedb` package is pure Python: it parses RelQL, plans, and
assembles contexts and token batches next to your data, and can score through
a cloud backend URL. This package adds the local alternative — `librt_c`, the
golden-verified C++ RT-J engine with its **native MiniLM text encoder** (all
text embedding runs in native code; there is no Python embedding path) —
behind the same scorer protocol:

```python
from relativedb import Engine
from relativedb_engine import RtNativeBackend

engine = Engine(schema, wiring, model_backend=RtNativeBackend(schema=schema))
```

It also carries the adaptation paths that are native by design:

- `Engine.fit_head` — frozen-backbone task heads, trained on the GPU
  (Metal or CUDA);
- `Engine.finetune` — full-checkpoint fine-tuning;
- `Engine.fit_xgboost` — the flat-feature tree backend
  (`pip install relativedb-engine[xgboost]`). Feature *derivation* happens in
  `relativedb.flat` (pure Python); this package evaluates the derived spec
  natively and drives XGBoost.

## Install

```bash
pip install relativedb-engine              # or: pip install relativedb[engine]
pip install relativedb-engine[xgboost]     # + the tree backend
```

Wheels bundle `librt_c` for macOS (universal2, 13.0+) and manylinux
x86_64/aarch64. From source, build `cpp/` with CMake and set
`RELATIVEDB_RT_LIB`. Checkpoints and the MiniLM snapshot resolve through the
Hugging Face cache on first use (`huggingface_hub` downloads them; the native
loaders are cache-first and never open a socket themselves).

## Serving instead

The same native engine runs as a web backend: build `cpp/` and start
`rt_serve --port 8500`, then point any number of light client processes at it
with `Engine(schema, wiring, model_backend="http://host:8500")`. The wire
carries prepared token batches with text as raw strings; the service embeds
and runs the forward.

## Development

```bash
pip install -e ../python -e ".[dev,xgboost]"
pytest -m "not integration"   # unit tier (needs librt_c, no checkpoint)
pytest -m integration         # + the real rt-j checkpoint and MiniLM snapshot
```

## License

Apache-2.0.
