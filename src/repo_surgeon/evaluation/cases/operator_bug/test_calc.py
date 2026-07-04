from calc import add


def test_add():
    assert add(2, 3) == 5


def test_add_zero():
    assert add(10, 0) == 10
