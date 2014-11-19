"""matrix-splitting iterations"""

import numpy as np
import math


def gs(x, A, b, **kwargs):
    """gauss-seidel"""
    d = np.diag(A)
    n = b.size
    
    error = 1

    omega = kwargs.get('omega', 1)
    
    old = np.copy(x)
    delta = np.zeros(n)
    
    while error > 0:

        for i in xrange(n):
            x[i] += omega * (b[i] - A[i, :].dot(x)) / d[i]

        delta[:] = x - old
        
        error = math.sqrt(delta.dot(delta))
        yield error
        old[:] = x




def jacobi(x, A, b, **kwargs):
    """jacobi"""

    d = np.diag(A)
    n = b.size

    omega = kwargs.get('omega', 2 / n)
    error = 1

    old = np.copy(x)
    delta = np.zeros(n)
    
    while error > 0:

        x += omega * (b - A.dot(x)) / d

        delta[:] = x - old
        
        error = math.sqrt(delta.dot(delta))
        yield error
        old[:] = x


def pgs(x, A, b, **kwargs):
    """projected gauss-seidel"""
    d = np.diag(A)
    n = b.size

    omega = kwargs.get('omega', 1)
    
    error = 1

    old = np.copy(x)
    delta = np.zeros(n)
    
    while error > 0:

        for i in xrange(n):
            x[i] += omega * (b[i] - A[i, :].dot(x)) / d[i]
            if x[i] < 0: x[i] = 0
            
        delta[:] = x - old
        
        error = math.sqrt(delta.dot(delta))
        yield error
        old[:] = x


def pjacobi(x, A, b, **kwargs):
    """projected jacobi"""

    d = np.diag(A)
    n = b.size

    omega = kwargs.get('omega', 2 / float(n))
    
    error = 1

    old = np.copy(x)
    delta = np.zeros(n)

    zero = np.zeros(n)
    
    while error > 0:

        x += omega * (b - A.dot(x)) / d
        x[:] = np.maximum(x, zero)
        
        delta[:] = x - old
        
        error = math.sqrt(delta.dot(delta))
        yield error
        old[:] = x




