"""Is this post about tech?

A lexicon, because the alternative — asking the model — needs labelled data to
be worth anything, and this file is where that labelled data comes from.

The lexicon is deliberately broader than the AI-only filter in the parent
directory: "tech stuff" for this community means software, hardware, ML, data,
security and the industry around them, not just LLMs.

``validate.py`` scores it against a hand-labelled sample and prints precision
and recall, so the number in the writeup is measured rather than assumed. Do not
quote a topic breakdown from this file without that number next to it.
"""
from __future__ import annotations

import re

TECH = re.compile(r"""\b(
    ai|a\.i\.|llms?|gpt-?\d*|claude|gemini|llama|mistral|qwen|deepseek|
    openai|anthropic|deepmind|huggingface|nvidia|

    machine[ -]learning|deep[ -]learning|neural|transformers?|
    embeddings?|inference|fine-?tun\w*|pretrain\w*|rlhf|
    diffusion|distillation|tokens?|prompt\w*|hallucinat\w*|agentic|agents?|
    benchmarks?|datasets?|gpus?|tpus?|compute|scaling[ -]laws?|
    neurips|icml|iclr|acl|emnlp|arxiv|preprints?|stoc|focs|soda|sigmod|
    # "model" is ambiguous (fashion, model organism, economic model), so it
    # only counts next to words that make it a computational one.
    (?:language|world|base|frontier|reasoning|larger?|small|open|closed|
       recurrent|generative|foundation|diffusion|reward|multimodal)[ -]models?|
    models?[ -](?:weights?|training|inference|card|collapse)|

    software|hardware|codebase|programming|developer|
    python|javascript|typescript|rust|golang|sql|linux|kernel|
    api|apis|sdk|cli|repo|repos|github|gitlab|open[ -]source|
    compiler|runtime|database|databases|server|servers|cloud|
    kubernetes|docker|latency|throughput|bandwidth|

    algorithm|algorithms|encryption|cryptograph\w*|
    # bare "security" also matches physical/national security, so require a
    # computing sense.
    (?:cyber|info|data|app|network|software)[ -]?security|
    security[ -](?:patch|flaw|researcher|team|update|advisory)|
    vulnerabilit\w*|exploit|malware|phishing|breach|zero-?day|
    privacy|surveillance|

    startup|startups|silicon[ -]valley|venture|vc|ipo|
    big[ -]tech|platform|platforms|apps|browser|
    chip|chips|semiconductor|data[ -]cent(?:er|re)s?|
    crypto|blockchain|bitcoin|

    tech|technolog\w*|computer|computing|internet|digital|automation|robot\w*
)\b""", re.IGNORECASE | re.VERBOSE)


def is_tech(text: str | None) -> bool:
    return bool(TECH.search(text or ""))
