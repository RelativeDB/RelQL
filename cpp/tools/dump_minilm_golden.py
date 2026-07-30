#!/usr/bin/env python
"""Regenerate the MiniLM conformance goldens for cpp/src/test_minilm.cpp.

Needs a venv with sentence-transformers (the dev/reference stack — production
never runs Python embedding). Writes cpp/testdata/minilm_golden.json:
inputs, HF token ids, and fp32 embeddings (raw + normalized) from
sentence-transformers/all-MiniLM-L12-v2 on CPU.
"""
import json
import pathlib

from sentence_transformers import SentenceTransformer

TEXTS = [
    "",
    "hello",
    "Hello, World!",
    "qty of orders",
    "order_date of orders",
    "signup_date of customers",
    "the quick brown fox jumps over the lazy dog",
    "SELECT * FROM users WHERE id = 42;",
    "naïve café — résumé",
    "Ünïcödé ştrîñgs",
    "日本語のテキスト",
    "混合 mixed 文字 text",
    "emoji 🚀 in the middle",
    "line\nbreaks\tand\ttabs",
    "  leading and trailing spaces  ",
    "ALL CAPS SENTENCE",
    "CamelCaseIdentifier",
    "snake_case_identifier",
    "1234567890",
    "3.14159 is pi, -42 is not",
    "user@example.com http://example.com/path?q=1",
    "double  spaces   collapse",
    "punctuation!!! everywhere??? really...",
    "'quoted' \"double quoted\"",
    "a",
    "ab",
    "supercalifragilisticexpialidocious",
    "pneumonoultramicroscopicsilicovolcanoconiosis words",
    "München Zürich København",
    "Владимир Прага Москва",
    "premium plan customer since 2019",
    "churned after 90 days of inactivity",
    "GitHub issue: segfault when loading quantized checkpoint on aarch64",
    "The customer placed 12 orders totaling $1,234.56 in Q3 2025.",
    "true false null",
    "active",
    "inactive",
    "label of issues",
    ("a very long text " * 40).strip(),   # > 128 word pieces, truncation
    "Mixed CASE with ümläuts and 中文 and 123 and !!!",
]


def main() -> None:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2",
                                device="cpu")
    tok = model.tokenizer
    raw = model.encode(TEXTS, normalize_embeddings=False,
                       show_progress_bar=False, convert_to_numpy=True)
    norm = model.encode(TEXTS, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)
    entries = []
    for i, t in enumerate(TEXTS):
        ids = tok(t, truncation=True, max_length=128)["input_ids"]
        entries.append({
            "text": t,
            "ids": ids,
            "raw": [float(x) for x in raw[i]],
            "norm": [float(x) for x in norm[i]],
        })
    out = pathlib.Path(__file__).resolve().parents[1] / "testdata"
    out.mkdir(exist_ok=True)
    (out / "minilm_golden.json").write_text(json.dumps(entries))
    print(f"wrote {out / 'minilm_golden.json'} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
