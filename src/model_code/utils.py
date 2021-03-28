"""
Some pure functions used frequently.
"""
import functools
from collections.abc import Iterable


def iterable(o):
    return isinstance(o, Iterable) and not isinstance(o, str)


def list_str(list):
    return [str(o) for o in list] if iterable(list) else str(list)


def _find_max_list(lists):
    return max(len(lst) for lst in lists)


def stretch(lists):
    """
    Stretch lists to same length.
    Shorter lists repeat to reach the same length as the longest.
    """
    lists = [lst if iterable(lst) else [lst] for lst in lists]
    max_len = _find_max_list(lists)
    return [(list(lst) * ((max_len - 1) // len(lst) + 1))[:max_len] for lst in lists]


def reduce_concat(x, sep=""):
    """
    Join a list into one string.
    """
    return functools.reduce(lambda x, y: str(x) + sep + str(y), x)


def paste(*lists, sep=" ", collapse=None):
    """
    Concatenate vectors after converting to character.
    The `Paste` function in R implementation in Python.

    Parameters
    ----------
    lists : array-like
        One or more objects, to be converted to character vectors.
    sep : string
        A character string to separate the terms
    collapse : string
        An optional character string to separate the results
    """
    lists = stretch(lists)
    result = map(lambda x: reduce_concat(x, sep=sep), zip(*lists))
    if collapse is not None:
        return reduce_concat(result, sep=collapse)
    return list(result)


def paste0(*lists, collapse=None):
    """
    paste0(..., collapse) is equivalent to paste(..., sep = "", collapse), slightly
    more efficiently.
    """
    return paste(*lists, sep="", collapse=collapse)
