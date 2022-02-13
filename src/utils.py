import numpy as np

def cached(f, cmp=lambda cache, data: cache == data):
    data = None
    def wrapper(*args, **kargs):
        nonlocal data
        if data is None or kargs.get('exec', False):
            data = f(*args, **kargs)
        return data
    return wrapper

def to_dictionary(characters):
    i2c = dict(enumerate(set(characters)))
    c2i = {v : k for (k, v) in i2c.items()}
    return c2i, i2c

def one_hot_vector(inp, vocab_size):
    # https://github.com/mcleonard/pytorch-charRNN/blob/master/utils.py#L41
    one_hot = np.zeros((inp.shape[0] * inp.shape[1], vocab_size), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), inp.flatten()] = 1
    one_hot = one_hot.reshape((*inp.shape, vocab_size))
    return one_hot