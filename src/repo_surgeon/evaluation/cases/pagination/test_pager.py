from pager import paginate


def test_page_has_exact_size():
    assert len(paginate(list(range(10)), 3)) == 3


def test_page_contents():
    assert paginate([1, 2, 3, 4, 5], 2) == [1, 2]
