import numpy as np
import sys

from functions.kconvolve import  kconvolve
from functions.arithmetic import *
from functions.general_max import general_max
from functions.general_min import general_min
from maptypes.raster  import  Raster
from maptypes.kernel import Kernel
Symbol = str
List = list
Number = (int, float)



class Procedure(object):
    "A user defined Scheme procedure."
    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env
    def __call__(self, *args):
        return eval(self.body, Env(self.parms, args, self.env))

class Env(dict):
    "An environment: a dict of {'var' : val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)



def tokenize(chars):
    return chars.replace('(', ' ( ').replace(')', ' ) ').split()

def parse(program):
    return read_from_tokens(tokenize(program))

def read_from_tokens(tokens):
    if len(tokens) == 0:
        raise SyntaxError('unexppected EOF while reading')
    token = tokens.pop(0)
    if '(' == token:
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0)
        # print L
        return L
    elif ')' == token:
        raise SyntaxError('unexpected )')
    else:
        return atom(token)

def atom(token):
    """ Number became Number, if string is double quoted it is 
    recongnized as Raster file or Kernel file based on whether
    it has .csv extension, other string is taken as Symbol.
    """
    try: return int(token)
    except ValueError:
        try: return float(token)
        except ValueError:
            if token[0] == '"' and token[-1] == '"':
                name = token[1:-1]
                try: return global_env.find(name)[name]
                except: 
                    if name[-4:] == '.csv':
                        return Kernel(name)
                    else:
                        return Raster(name)
            else:
                return Symbol(token)
            

def standard_env():
    "An environment with some Scheme standard procedures."
    import math, operator as op
    env = Env()
    env.update(vars(math)) # sin, cos, sqrt, pi, ...
    env.update({
        '+':add, '-':sub, '*':mul, '/':div, 
        '>':op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '=':op.eq, 
        'abs':     abs,
        'apply':   apply,
        'begin':   lambda *x: x[-1],
        'car':     lambda x: x[0],
        'cdr':     lambda x: x[1:], 
        'cons':    lambda x,y: [x] + y,
        'eq?':     op.is_, 
        'equal?':  op.eq, 
        'length':  len, 
        'list':    lambda *x: list(x), 
        'list?':   lambda x: isinstance(x,list), 
        'map':     map,
        'max':     general_max,
        'min':     general_min,
        'not':     op.not_,
        'null?':   lambda x: x == [], 
        'number?': lambda x: isinstance(x, Number),   
        'procedure?': callable,
        'round':   round,
        'symbol?': lambda x: isinstance(x, Symbol),
        'kconvolve': kconvolve
    })
    return env

global_env = standard_env()

def eval(x, env=global_env): 
    if isinstance(x, Symbol):
        return env.find(x)[x] 
    elif not isinstance(x, List):
        return x 
    elif x[0] == 'quote':
        (_, exp) = x
        return exp
    elif x[0] == 'if':
        (_, test, conseq, alt) = x
        exp = (conseq if eval(test, env) else alt)
        return eval(exp, env)
    elif x[0] == 'define':
        (_, var, exp) = x
        result = eval(exp, env)
        if  isinstance(var, Raster):            
            var.write_data(result.data, 0, 0, result.nodata, result.x_size, result.y_size, 
                           result.driver, result.georef, result.proj)
            env[var.name] = var
            env[var] = var
        else:
            env[var] = result
    elif x[0] == 'set!':
        (_, var, exp) = x
        result = eval(exp, env)
        env.find(var)[var] = result

    elif x[0] == 'lambda':
        (_, parms, body) = x
        return Procedure(parms, body, env)
    else:
        proc = eval(x[0], env)
        args = [eval(arg, env) for arg in x[1:]]
        return proc(*args)


