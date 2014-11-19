"""LCP related tools"""

import numpy as np
import metric

def load(filename):
    """load lcp from file"""
    with open(filename) as f:
        n = int(f.readline())

        M = np.zeros((n, n))
        for i in xrange(n):
            M[i, :] = map(float, f.readline().split())

        q = np.zeros(n)
        q[:] = map(float, f.readline().split())

        return M, q




from itertools import izip

def bench(lcp, solver, **kwargs):
    """benchmark a lcp solver"""
    
    n = lcp[1].size

    iterations = kwargs.get('iterations', 100)
    precision = kwargs.get('precision', 0)
    error = kwargs.get('metric', metric.residual_norm( lcp ) )

    x = kwargs.get('initial', np.zeros(n))

    for (k, info) in izip(xrange(iterations),
                          solver(x, lcp)):

        e = error(x)
        yield k, e
        
        if e <= precision: break


