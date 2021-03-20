from src.model_code.utils import list_str
from src.model_code.utils import paste
from src.model_code.utils import stretch


def test_list_str():
    assert ["1", "2", "3"] == list_str([1, 2, 3])


def test_stretch():
    lists = [[1, 2], [1, 2, 3, 4], 1, [1, 2, 3]]
    expect = [[1, 2, 1, 2], [1, 2, 3, 4], [1, 1, 1, 1], [1, 2, 3, 1]]
    assert expect == stretch(lists)


def test_paste():
    a = "beta"
    b = ["a", "b", "c"]
    c = range(5)
    expect = ["beta.a.0", "beta.b.1", "beta.c.2", "beta.a.3", "beta.b.4"]
    assert paste(a, b, c, sep=".") == expect
