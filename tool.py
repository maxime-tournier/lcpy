"""LCP related tools"""

import numpy as np
import metric

import gzip

def load_lcp(filename):
    """load lcp from file"""

    ext = filename.split('.')[-1]

    import gzip
    method = gzip.open if ext == 'gz' else open
    
    with method(filename) as f:
        n = int(f.readline())

        M = np.zeros((n, n))
        for i in xrange(n):
            M[i, :] = map(float, f.readline().split())

        q = np.zeros(n)
        q[:] = map(float, f.readline().split())

        return M, q

def load_vec(filename):
    ext = filename.split('.')[-1]

    import gzip
    method = gzip.open if ext == 'gz' else open

    with method(filename) as f:
        return np.array( map(float, f.readline().split()) )
    

from itertools import izip

def bench(lcp, solver, **kwargs):
    """benchmark a lcp solver"""
    
    n = lcp[1].size

    iterations = kwargs.get('iterations', 100)
    precision = kwargs.get('precision', 0)
    error = kwargs.get('metric', metric.residual_norm( lcp ) )

    x = np.copy( kwargs.get('initial', np.zeros(n)) )

    yield 0, error(x)
    
    for (k, info) in izip(xrange(iterations),
                          solver(x, lcp)):

        e = error(x)
        yield k + 1, e
        
        if e <= precision: break


