from store import lookup


def test_missing_key_returns_none():
    assert lookup({}, "x") is None


def test_present_key_returns_value():
    assert lookup({"a": 1}, "a") == 1
