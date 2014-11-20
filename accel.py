"""accelerations for LCP algorithms"""

import numpy as np
import math


from itertools import izip

def crossing(x, y):
    return  ( (-xi / (yi - xi)) if (xi - yi != 0) else 1 for xi, yi in izip(x, y) )


def alpha(x, y):
    func = lambda x, y: y if (y > 0 and y < x) else x
    return reduce(func, crossing(x, y), 1)


def nlnscg( solver, **kwargs ):

    def res(x, (M, q)):
        n = q.size

        d = kwargs.get('metric', np.ones(n) )

        grad = np.zeros(n)
        p = np.zeros(n)
        old = np.zeros(n)
        grad2 = 1

        omega = kwargs.get('omega', 1)
        
        sub = solver(x, (M, q))
        
        old[:] = x
        for error in sub:
            yield error
            
            grad[:] = old - x
            old_grad2 = grad2
            
            grad2 = grad.dot( d * grad)
            if grad2 == 0: break
            
            beta = grad2 / old_grad2

            if beta > 1:
                p[:] = np.zeros(n)
            else:
                p[:] = beta * p - grad
                x[:] = old + omega * p
                                
                
            old[:] = x

    
    res.__doc__ = "NLNSCG{} + {}".format( '*' if 'metric' in kwargs else '',
                                          solver.__doc__ )
    
    return res


def anderson( solver, m = 4, **kwargs ):
    
    def res(x, (M, q)):
        n = q.size

        sub = solver(x, (M, q))

        g = np.zeros( (n, m) )
        f = np.zeros( (n, m) )
        k = np.zeros( (m, m) )

        tmp = np.zeros(m)
        ones = np.ones(m)
        delta = np.zeros(n)
        
        d = kwargs.get('metric', np.ones(n) )
        reset = kwargs.get('reset', False)
        
        index = 0

        g[:, m - 1] = x        
        for error in sub:
            yield error

            prev = (index + m - 1) % m
            delta[:] = x - g[:, prev]

            if reset and delta.dot(delta) > f[:, prev].dot(f[:, prev]):
                g = np.zeros( (n, m) )
                f = np.zeros( (n, m) )
                k = np.zeros( (m, m) )
                
            g[:, index] = x
            f[:, index] = delta

            k[:, index] = f.transpose().dot( d * f[:, index])
            k[index, :] = k[:, index].transpose()
            
            tmp[:] = np.linalg.lstsq(k, ones)[0]
            tmp /= sum(tmp)
            
            delta[:] = g.dot(tmp)

            if reset:
                a = alpha(x, delta)
                x[:] = (1 - a) * x + a * delta
            else:
                x[:] = delta
                
            index = (index + 1) % m
            
    res.__doc__ = "Anderson{}({}) + {}".format('*' if 'metric' in kwargs else '',
                                               m,
                                               solver.__doc__)
    return res
