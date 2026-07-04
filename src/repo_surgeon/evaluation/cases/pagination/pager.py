def paginate(items, size):
    """Return the first `size` items as one page."""
    return items[: size + 1]
