# relativedb-engine

The optional in-process inference engine for
[relativedb](https://pypi.org/project/relativedb/).

The base `relativedb` package is pure Python: it parses RelQL, plans, and
assembles contexts and token batches next to your data, and can score through
a cloud backend URL. This package adds the local alternative over the shared
[relational-transformers](https://relationaltransformers.com) runtime: Triton
FP16 inference by default on CUDA, torch on MPS and CPU, ONNX for exported
graphs, and MiniLM text encoding in torch. Everything sits behind the same
scorer protocol:

```python
from relativedb import Engine
from relativedb_engine import RtBackend

engine = Engine(schema, wiring, model_backend=RtBackend(schema=schema))
```

It also carries the adaptation paths:

- `Engine.fit_head` — frozen-backbone task heads, trained with torch AdamW;
- `Engine.finetune` — full-checkpoint fine-tuning through torch.

## Install

```bash
pip install relativedb-engine              # or: pip install relativedb[engine]
pip install "relativedb-engine[triton]"    # primary CUDA inference
```

Checkpoints and the MiniLM snapshot resolve through the Hugging Face cache on
first use (`huggingface_hub` downloads them; the loaders are cache-first and
never open a socket themselves).

## Serving

The primary CUDA worker speaks the existing remote scorer protocol:

```bash
pip install "relativedb-engine[triton]"
rt_triton_serve --checkpoint /models/model.safetensors --port 8500
```

Point any number of light client processes at it with
`Engine(schema, wiring, model_backend="https://worker.example")`. The wire
carries prepared token batches with text as raw strings; the worker embeds
them next to the model and serializes concurrent forwards over its reusable
Triton buffers.

## Development

```bash
pip install -e ../python -e ".[dev]"
pytest -m "not integration"   # unit tier (no checkpoint download)
pytest -m integration         # + the real rt-j checkpoint and MiniLM snapshot
```

## License

Apache-2.0.
