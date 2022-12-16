'''
Einops exts
#@ FROM https://github.com/lucidrains/einops-exts
'''

import re
from torch import nn
from functools import wraps, partial

from einops import rearrange, reduce, repeat

# checking shape
# @nils-werner
# https://github.com/arogozhnikov/einops/issues/168#issuecomment-1042933838

def check_shape(tensor, pattern, **kwargs):
    return rearrange(tensor, f"{pattern} -> {pattern}", **kwargs)

# do same einops operations on a list of tensors

def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)
    return inner

# do einops with unflattening of anonymously named dimensions
# (...flattened) ->  ...flattened

def _with_anon_dims(fn):
    @wraps(fn)
    def inner(tensor, pattern, **kwargs):
        regex = r'(\.\.\.[a-zA-Z]+)'
        matches = re.findall(regex, pattern)
        get_anon_dim_name = lambda t: t.lstrip('...')
        dim_prefixes = tuple(map(get_anon_dim_name, set(matches)))

        update_kwargs_dict = dict()

        for prefix in dim_prefixes:
            assert prefix in kwargs, f'dimension list "{prefix}" was not passed in'
            dim_list = kwargs[prefix]
            assert isinstance(dim_list, (list, tuple)), f'dimension list "{prefix}" needs to be a tuple of list of dimensions'
            dim_names = list(map(lambda ind: f'{prefix}{ind}', range(len(dim_list))))
            update_kwargs_dict[prefix] = dict(zip(dim_names, dim_list))

        def sub_with_anonymous_dims(t):
            dim_name_prefix = get_anon_dim_name(t.groups()[0])
            return ' '.join(update_kwargs_dict[dim_name_prefix].keys())

        pattern_new = re.sub(regex, sub_with_anonymous_dims, pattern)

        for prefix, update_dict in update_kwargs_dict.items():
            del kwargs[prefix]
            kwargs.update(update_dict)

        return fn(tensor, pattern_new, **kwargs)
    return inner

# generate all helper functions

rearrange_many = _many(rearrange)
repeat_many = _many(repeat)
reduce_many = _many(reduce)

rearrange_with_anon_dims = _with_anon_dims(rearrange)
repeat_with_anon_dims = _with_anon_dims(repeat)
reduce_with_anon_dims = _with_anon_dims(reduce)