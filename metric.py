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


def lcp_error( (M, q) ):
    """primal + dual + complementarity error norms"""

    n = q.size

    primal = np.zeros(n)
    dual = np.zeros(n)
    
    zero = np.zeros(n)
    
    def res(x):
        """primal + dual + complementarity error norms"""
        primal[:] = q + M.dot(x)

        compl = math.fabs( primal.dot( x ) )
        
        primal[:] = np.minimum( primal, zero )
        dual[:] = np.minimum( x, zero )

        return math.sqrt(primal.dot(primal)) + math.sqrt(dual.dot(dual)) + math.sqrt( compl )

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
        
        r[:] = q + M.dot(x)
        m[:] = np.minimum(x, r)
        
        return math.sqrt(m.dot(d * m))

    res.__doc__ = """primal/dual minimum norm{}""".format('*' if 'metric' in kwargs else '' )
    return res
