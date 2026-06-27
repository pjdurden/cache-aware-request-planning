from ccrp.canonicalize import canonical_prefix


def test_key_order_does_not_change_canonical_form():
    a = canonical_prefix([{"name": "f", "params": {"x": 1, "y": 2}}])
    b = canonical_prefix([{"params": {"y": 2, "x": 1}, "name": "f"}])
    assert a == b


def test_whitespace_is_normalized_in_text_parts():
    a = canonical_prefix(["You   are\n a   bot"])
    b = canonical_prefix(["You are a bot"])
    assert a == b


def test_different_content_differs():
    a = canonical_prefix(["system A", {"name": "f"}])
    b = canonical_prefix(["system B", {"name": "f"}])
    assert a != b
