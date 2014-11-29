import numpy as np
import math

# metics
def residual_norm( (M, q) ):
    """residual norm metric"""
    n = q.size
    r = np.zeros(n)

    def res(x):
        """residual norm"""
        r[:] = q + M.dot(x)
        return math.sqrt(r.dot(r))

    return res


def energy_norm( (M, q), lower_bound = 0 ):
    """energy norm metric"""
    n = q.size()
    r = np.zeros(n)

    def res(x):
        """energy norm"""
        r[:] = q + 0.5 * M.dot(x)
        return math.sqrt(x.dot(r) - lower_bound)

    return res


def lcp_error( (M, q), **kwargs):
    """primal + dual + complementarity error norms"""

    n = q.size

    p = np.zeros(n)
    d = np.zeros(n)
    
    zero = np.zeros(n)

    primal = kwargs.get('primal', True)
    dual = kwargs.get('dual', True)
    compl = kwargs.get('compl', True)
    
    def res(x):
        p[:] = q + M.dot(x)

        c = math.fabs( p.dot( x ) )

        p[:] = np.minimum( p, zero )
        d[:] = np.minimum( x, zero )

        s = 0
        if primal: s += math.sqrt(p.dot(p))
        if dual: s += math.sqrt(d.dot(d))
        if compl: s += math.sqrt( c )

        return s

    res.__doc__ = "{}{}{} error norms".format( 'primal' if primal else '',
                                                 ' + dual' if dual else '',
                                                 ' + complementarity' if compl else '' )

    return res



def minimum_norm( (M, q), **kwargs ):
    """
    a positive convex merit function for LCP: norm( min(Mx + q, x) )
    
    it is zero if and only if the LCP is solved

    """
    
    n = q.size
    r = np.zeros(n)
    m = np.zeros(n)
    d = kwargs.get('metric', np.ones(n))
    
    def res(x):
        
        r[:] = (q + M.dot(x)) 
        m[:] = np.minimum(x * d, r)
        
        return math.sqrt(m.dot(m))

    res.__doc__ = """primal/dual minimum norm{}""".format('*' if 'metric' in kwargs else '' )
    return res
