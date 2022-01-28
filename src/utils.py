def cached(f, cmp=lambda cache, data: cache == data):
    data = None
    def wrapper(*args, **kargs):
        nonlocal data
        if data is None or kargs.get('exec', False):
            data = f(*args, **kargs)
        return data
    return wrapper