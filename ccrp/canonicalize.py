import json


def _normalize_text(s):
    return " ".join(s.split())


def canonical_prefix(parts):
    """Stable string identity for a cacheable prefix.

    Sorts dict keys (kills nondeterministic JSON ordering) and collapses
    whitespace in text parts (kills incidental formatting differences).
    """
    chunks = []
    for part in parts:
        if isinstance(part, str):
            chunks.append("T:" + _normalize_text(part))
        else:
            chunks.append("J:" + json.dumps(part, sort_keys=True, separators=(",", ":")))
    return "\n".join(chunks)
