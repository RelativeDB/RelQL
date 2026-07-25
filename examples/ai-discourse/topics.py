"""An optional, deliberately crude topical filter.

The universe is "what the AI research community posted today", which is the
right frame most of the time: a complaint about NeurIPS reviewing is an AI
discussion even though it never says "AI". But the community also contains
science and tech journalists, and some days the post that travels furthest is
a bear opening a freezer.

``--ai-only`` narrows to posts whose text mentions something explicitly AI. Be
aware of what it costs: on the shipped snapshot it keeps 27% of the universe
and it throws away real AI-community content that happens not to use the
vocabulary. It is a keyword filter and it behaves like one.

The better version of this does not live here. RelQL can classify the topic
with the same model that does the ranking —

    PREDICT posts.topic FROM posts WHERE posts.topic IS NULL

— the auto-labelling pattern from the top-level README, seeded with a few
hand-labelled posts. That reads the text instead of matching it. It is left
out of the measured results because this snapshot has no hand-labelled topics
to score it against, and an unevaluated classifier is decoration.
"""
from __future__ import annotations

import re

TERMS = re.compile(
    r"\b(ai|a\.i\.|llms?|gpt|claude|gemini|openai|anthropic|deepmind|"
    r"transformers?|neural|machine learning|deep learning|nlp|agentic|agents?|"
    r"models?|training|fine-?tun\w*|inference|benchmarks?|datasets?|chatbot|"
    r"rlhf|diffusion|embeddings?|tokens?|prompt\w*|hallucinat\w*|alignment|"
    r"agi|open-?weights?|neurips|icml|iclr|acl|arxiv|gpus?|compute|scaling)\b",
    re.IGNORECASE)


def is_ai(text: str | None) -> bool:
    return bool(TERMS.search(text or ""))


def filter_posts(posts: list[dict], enabled: bool) -> list[dict]:
    return [p for p in posts if is_ai(p.get("text"))] if enabled else posts
