"""accelerations for LCP algorithms"""

import numpy as np
import math

def nlnscg( solver ):

    def res(x, (M, q)):
        n = q.size

        grad = np.zeros(n)
        p = np.zeros(n)
        
        run = solver(x, (M, q))

        old = np.copy(x)
        yield next(run)
        grad[:] = old - x
        grad2 = grad.dot(grad)

        error = math.sqrt( grad2 )
        if error == 0: return
        
        old[:] = x
        for _ in run:
            
            grad[:] = old - x

            old_grad2 = grad2
            grad2 = grad.dot(grad)

            error = math.sqrt( grad2 )
            if error == 0: return

            beta = grad2 / old_grad2

            if beta > 1:
                p[:] = np.zeros(n)
            else:
                x += beta * p
                p[:] = beta * p - grad

            yield error

            old[:] = x

    res.__doc__ = "NLNSCG + " + solver.__doc__
    return res
