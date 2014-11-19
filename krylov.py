"""krylov methods"""

import numpy as np
import math


def cg(x, A, b):
    """conjugate gradient"""
    n = b.size
    
    r = b - A.dot(x)
    p = np.copy(r)

    r2 = r.dot(r)
    error = math.sqrt(r2)

    Ap = np.zeros(n)
    
    while error > 0:
        Ap[:] = A.dot(p)
        alpha = r2 / p.dot(Ap)

        x += alpha * p
        r -= alpha * Ap

        old_r2 = r2
        r2 = r.dot(r)
        
        error = math.sqrt(r2)
        
        yield error, r

        beta = r2 / old_r2
        p[:] = r + beta * p
    


def cr(x, A, b):
    """conjugate residual"""
    
    r = b - A.dot(x)
    p = np.copy(r)

    Ar = A.dot(r)
    Ap = np.copy(Ar)

    rAr = r.dot(Ar)
    r2 = r.dot(r)

    error = math.sqrt(r2)
    
    while error > 0:
        alpha = rAr / Ap.dot(Ap)
        
        x += alpha * p
        r -= alpha * Ap

        r2 = r.dot(r)
        error = math.sqrt(r2)
        
        yield error, r

        Ar[:] = A.dot(r)
        old_rAr = rAr

        rAr = r.dot(Ar)
        
        beta = rAr / old_rAr
        
        p[:] = r + beta * p
        Ap[:] = Ar + beta * Ap



# TODO pcg, pcr

