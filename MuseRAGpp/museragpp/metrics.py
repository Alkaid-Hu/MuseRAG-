import re
def citation_coverage(answer: str) -> float:
    sents = [s for s in re.split(r"[。.!?]\s*", answer) if s.strip()]
    if not sents: return 0.0
    cited = sum(1 for s in sents if re.search(r"\[S\d+\]", s))
    return round(cited / len(sents), 3)