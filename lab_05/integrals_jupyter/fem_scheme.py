from sympy import *

x, y, h, x_k, y_k, varphi_ik = symbols('x y h x_k y_k varphi_ik')

fem = {
    'self_self': (
        {
            'first': {
                'name': 'delta1',
                'subs': []
            },
            'second': {
                'name': 'delta1',
                'subs': []
            },
        },
        {
            'first': {
                'name': 'delta2',
                'subs': []
            },
            'second': {
                'name': 'delta2',
                'subs': []
            },
        },
        {
            'first': {
                'name': 'delta3',
                'subs': []
            },
            'second': {
                'name': 'delta3',
                'subs': []
            },
        },
        {
            'first': {
                'name': 'delta4',
                'subs': []
            },
            'second': {
                'name': 'delta4',
                'subs': []
            },
        },
        {
            'first': {
                'name': 'delta5',
                'subs': []
            },
            'second': {
                'name': 'delta5',
                'subs': []
            },
        },
        {
            'first': {
                'name': 'delta6',
                'subs': []
            },
            'second': {
                'name': 'delta6',
                'subs': []
            },
        },
    ),
    'right': (
        {
            'first': {
                'name': 'delta1',
                'subs': []
            },
            'second': {
                'name': 'delta3',
                'subs': [(x_k, x_k + h)]
            },
        },
        {
            'first': {
                'name': 'delta6',
                'subs': []
            },
            'second': {
                'name': 'delta4',
                'subs': [(x_k, x_k + h)]
            },
        },
    ),
    'right_top': (
        {
            'first': {
                'name': 'delta1',
                'subs': []
            },
            'second': {
                'name': 'delta5',
                'subs': [(x_k, x_k + h), (y_k, y_k + h)]
            },
        },
        {
            'first': {
                'name': 'delta2',
                'subs': []
            },
            'second': {
                'name': 'delta4',
                'subs': [(x_k, x_k + h), (y_k, y_k + h)]
            },
        },
    ),
    'top': (
        {
            'first': {
                'name': 'delta2',
                'subs': []
            },
            'second': {
                'name': 'delta6',
                'subs': [(y_k, y_k + h)]
            },
        },
        {
            'first': {
                'name': 'delta3',
                'subs': []
            },
            'second': {
                'name': 'delta5',
                'subs': [(y_k, y_k + h)]
            },
        },
    ),

    'left': (
        {
            'first': {
                'name': 'delta3',
                'subs': []
            },
            'second': {
                'name': 'delta1',
                'subs': [(x_k, x_k - h)]
            },
        },
        {
            'first': {
                'name': 'delta4',
                'subs': []
            },
            'second': {
                'name': 'delta6',
                'subs': [(x_k, x_k - h)]
            },
        },
    ),
    'left_bottom': (
        {
            'first': {
                'name': 'delta4',
                'subs': []
            },
            'second': {
                'name': 'delta2',
                'subs': [(x_k, x_k - h), (y_k, y_k - h)]
            },
        },
        {
            'first': {
                'name': 'delta5',
                'subs': []
            },
            'second': {
                'name': 'delta1',
                'subs': [(x_k, x_k - h), (y_k, y_k - h)]
            },
        },
    ),
    'bottom': (
        {
            'first': {
                'name': 'delta6',
                'subs': []
            },
            'second': {
                'name': 'delta2',
                'subs': [(y_k, y_k - h)]
            },
        },
        {
            'first': {
                'name': 'delta5',
                'subs': []
            },
            'second': {
                'name': 'delta3',
                'subs': [(y_k, y_k - h)]
            },
        },
    )
}
