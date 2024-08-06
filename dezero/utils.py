def _dot_var(v, vervose=False):
    dot_var = '{} [labels="{}", color=orange, style=filled]\n'
    name = '' if v.name is None else v.name
    if vervose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)

def _dot_func(f):
    dot_func = '{} [labels="{}", color=orange, style=filled]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)
    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
        