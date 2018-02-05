"""accelerations for LCP algorithms"""

import numpy as np
import math


from itertools import izip

def crossing(x, y):
    return  ( (-xi / (yi - xi)) if (xi - yi != 0) else 1 for xi, yi in izip(x, y) )


def max_step(x, y):
    """maximum step from x to y remaining positive"""

    alpha = 1.0
    index = 0
    
    for i, (xi, yi) in enumerate(zip(x, y)):
        if xi == yi: continue
        value = -xi / (yi - xi)
        if alpha > value > 0:
            alpha = value
            index = i

    return index, alpha
    
    # func = lambda (ix, x), (iy, y): (iy, y) if (y > 0 and y < x) else (ix, x)
    # return reduce(func, enumerate(crossing(x, y)), (0, 1.0) )

def norm(x): return math.sqrt(x.dot(x))



def cr_pjacobi( x, (M, q), **kwargs ):
    """CR-PJacobi"""
    n = q.size

    z = np.zeros(n)
    r = np.zeros(n)
    p = np.zeros(n)

    Mx = np.zeros(n)
    Mz = np.zeros(n)
    Mp = np.zeros(n)
    mask = np.zeros(n)
    y = np.zeros(n)

    d = kwargs.get('diag', np.array(np.diag(M)) )

    omega = kwargs.get('omega', 2.0 / n)

    old_x = np.zeros(n)
    old_Mx = np.zeros(n)
    zMz = 1

    beta = 1
    k = 0

    Mx[:] = M.dot(x)

    def restart():
        # print k, 'restart'
        return np.zeros(n), np.zeros(n)

    while True:

        # backup
        old_x[:] = x
        old_Mx[:] = Mx

        # jacobi
        x -= omega * (q + Mx) / d
        mask = x > 0 

        # projection
        x[:] = mask * x

        yield

        # active-set change during jacobi ?
        if any( mask != (old_x > 0) ):
            p[:], Mp[:] = restart()
            Mx[:] = M.dot(x)
        else:
            # preconditioned gradient
            z[:] = (x - old_x) / omega
            
            # this is the only matrix-vector product. note: it is
            # *much* more numerically stable to compute Mx from Mz
            # than the contrary
            Mz[:] = M.dot(z)
            Mx[:] += omega * Mz
            
            old_zMz = zMz
            zMz = z.dot( Mz )

            if zMz <= 0:
                break

            beta = zMz / old_zMz
            
            p[:] = beta * p + z
            Mp[:] = beta * Mp + Mz

            y[:] = mask * Mp / d
            alpha = zMz / Mp.dot(y)

            y[:] = old_x + alpha * p
            index, step = max_step(x, y)

            x[:] += step * (y - x)
            Mx[:] += step * (alpha * Mp - omega * Mz)
            
            # # sanity check:
            # y[:] = M.dot(x)
            # print norm(y - Mx)
            
            if step < 1:
                p[:], Mp[:] = restart()
                # p[:], Mp[:] = z, Mz
                # print k, 'step', step
            
        k += 1
    print 'breakdown, zMz:', zMz






def cg_pjacobi( x, (M, q), **kwargs ):
    """CG-PJacobi"""
    n = q.size

    z = np.zeros(n)
    r = np.zeros(n)
    p = np.zeros(n)

    Mx = np.zeros(n)
    Mz = np.zeros(n)
    Mp = np.zeros(n)
    mask = np.zeros(n)
    y = np.zeros(n)

    d = kwargs.get('diag', np.array(np.diag(M)) )

    if 'diag' in kwargs:
        print 'using mass-splitting prec'

    omega = kwargs.get('omega', 2.0 / n)

    old_x = np.zeros(n)
    old_Mx = np.zeros(n)
    zr = 1

    beta = 1
    k = 0

    Mx[:] = M.dot(x)

    def restart():
        # print k, 'restart'
        return np.zeros(n), np.zeros(n)

    while True:

        # backup
        old_x[:] = x
        old_Mx[:] = Mx

        # jacobi
        x -= omega * (q + Mx) / d
        mask = x > 0 

        # projection
        x[:] = mask * x

        yield

        # active-set change during jacobi ?
        if any( mask != (old_x > 0) ):
            p[:], Mp[:] = restart()
            Mx[:] = M.dot(x)
        else:
            # preconditioned gradient
            z[:] = (x - old_x) / omega

            # gradient
            r[:] = d * z

            # this is the only matrix-vector product. note: it is
            # *much* more numerically stable to compute Mx from Mz
            # than the contrary
            Mz[:] = M.dot(z)
            Mx[:] += omega * Mz
            
            old_zr = zr
            zr = z.dot(r)

            if zr <= 0: break

            # reorthogonalize yo
            # bob = d * r
            # p -=  (p.dot( bob ) / (r.dot(bob) )) * r
            
            beta = zr / old_zr
            
            p[:] = beta * p + z
            Mp[:] = beta * Mp + Mz
            # Mp[:] = M.dot(p)
            
            alpha = zr / p.dot(Mp)
            
            y[:] = old_x + alpha * p
            index, step = max_step(x, y)

            x[:] += step * (y - x)
            Mx[:] += step * (alpha * Mp - omega * Mz)
            
            # # sanity check:
            # y[:] = M.dot(x)
            # print norm(y - Mx)
            
            if step < 1:
                p[:], Mp[:] = restart()
                pass
                # print k, 'step', step
            
        k += 1
    print 'breakdown, zr:', zr



def doubidou( x, (M, q), **kwargs ):
    """doubidou"""
    n = q.size

    z = np.zeros(n)
    r = np.zeros(n)
    p = np.zeros(n)

    Mx = np.zeros(n)
    Mz = np.zeros(n)
    Mp = np.zeros(n)
    mask = np.zeros(n)
    y = np.zeros(n)

    d = kwargs.get('diag', np.diag(M) )
    omega = kwargs.get('omega', 2.0 / n)

    old_x = np.zeros(n)
    old_Mx = np.zeros(n)

    zMz = 1
    k = 0

    Mx[:] = M.dot(x)

    def restart():
        # print k, 'restart'
        return np.zeros(n), np.zeros(n)

    while zMz > 0:

        # backup
        old_x[:] = x
        old_Mx[:] = Mx

        # jacobi
        x -= omega * (q + Mx) / d
        mask = x > 0 

        # projection
        x[:] = mask * x

        yield
        
        # preconditioned gradient
        z[:] = (x - old_x) / omega
        
        # this is the only matrix-vector product. note: it is
        # *much* more numerically stable to compute Mx from Mz
        # than the contrary
        Mz[:] = M.dot(z)
        Mx[:] += omega * Mz
        
        old_zMz = zMz
        zMz = z.dot( Mz )
        # zMz = z.dot( d * z)
        
        if zMz <= 0: break

        beta = zMz / old_zMz

        if beta > 1:
            p[:], Mp[:] = restart()
        else:
            p[:] = beta * p + z
            Mp[:] = beta * Mp + Mz
            
            x[:] = old_x + omega * p
            Mx[:] = old_Mx + omega * Mp
            
        k += 1


def nlnscg_ls( solver, **kwargs ):

    def res(x, (M, q)):
        n = q.size

        z = np.zeros(n)
        p = np.zeros(n)
        old = np.zeros(n)

        diag = kwargs.get('diag', np.diag(M))
        
        z2 = 1

        sub = solver(x, (M, q), **kwargs)

        old[:] = x

        for error in sub:
            yield error
            
            z[:] = (x - old)
            old_z2 = z2

            z2 = z.dot(diag * z)

            if z2 == 0: break
            
            beta = z2 / old_z2

            if beta > 1:
                p[:] = np.zeros(n)
            else:
                p[:] = beta * p + z
                x[:] = old + p

                f_old = (old > 0) * (M.dot(old) + q)
                f_new = (x > 0) * (M.dot(x) + q)
            
                df = (f_new - f_old)
                eps = 0
                a = (eps - f_old.dot(df)) / (eps + df.dot(df))
                
                x[:] = old + a * p
                
            old[:] = x
    
    res.__doc__ = "NLNSCG + {} ".format( solver.__doc__ )
    
    return res



def nlnscg( solver, **kwargs ):

    def res(x, (M, q)):
        n = q.size

        z = np.zeros(n)
        p = np.zeros(n)
        old = np.zeros(n)

        d = kwargs.get('diag', np.ones(n))

        z2 = 1
        
        sub = solver(x, (M, q), **kwargs)

        old[:] = x

        for error in sub:
            z[:] = (x - old)

            old_z2 = z2
            z2 = z.dot( d * z )

            if z2 == 0: break
            
            beta = z2 / old_z2

            if beta > 1:
                p[:] = np.zeros(n)
            else:
                p[:] = beta * p + z
                x[:] = old + p
                    
            old[:] = x
            yield

    res.__doc__ = "NLNSCG + {} ".format( solver.__doc__ )
    
    return res




def andy( x, (M, q), **kwargs):
    """andy"""
    
    n = q.size
    m = 2

    old_x = np.zeros(n)

    g = np.zeros(n)
    old_g = np.zeros(n)

    Mg = np.zeros(n)
    old_Mg = np.zeros(n)

    Mx = np.zeros(n)
    old_Mx = np.zeros(n)

    f = np.zeros(n)
    old_f = np.zeros(n)
    df = np.zeros(n)
    
    d = kwargs.get('diag', np.diag(M))
    omega = kwargs.get('omega', 2.0 / n)

    eps = kwargs.get('eps', 0.0)
    
    Mx[:] = M.dot(x)

    while True:

        # jacobi
        x[:] -= omega * (q + Mx) / d

        # projection
        x *= (x > 0)

        yield 

        # backup jacobi point
        old_g[:] = g
        g[:] = x

        Mx[:] = M.dot(x)

        old_Mg[:] = Mg
        Mg[:] = Mx
        
        # anderson
        old_f[:] = f
        
        f[:] = (x > 0) * (q + Mx)
        # f[:] = np.minimum(x, Mx + q)

        df[:] = f - old_f

        df2 = df.dot(df)
        # if df2 == 0: break
        alpha = 1 if df2 == 0 else (eps - old_f.dot(df)) / (eps + df2)
        
        x[:] = old_g + alpha * ( g - old_g )
        Mx[:] = old_Mg + alpha * (Mg - old_Mg)
        
        
def anderson( solver, **kwargs ):

    m = kwargs.get('m', 1.0 )
    
    def res(x, (M, q)):
        n = q.size

        sub = solver(x, (M, q), **kwargs)
        
        g = np.zeros( (n, m) )
        f = np.zeros( (n, m) )
        k = np.zeros( (m, m) )

        tmp = np.zeros(m)
        ones = np.ones(m)
        delta = np.zeros(n)
        
        d = kwargs.get('metric', np.ones(n) )
        reset = kwargs.get('reset', False)
        omega = kwargs.get('omega', 1.0)
        
        diag = np.diag(M)
        
        index = 0

        g[:, m - 1] = x

        flag = False
        skip = False
        diag_sqrt = np.sqrt( diag )
        zob2 = 1
        
        it = 0
        for error in sub:
            yield error

            prev = (index + m - 1) % m
            x_prev = g[:, prev]

            primal = (M.dot(x) + q)
            
            # mask = (x > 0) * (primal < 0)
            mask = (x > 0)
            # mask = primal < 0
            # mask = (primal < 0) * (x > 0)

            delta[:] = mask * primal

            # print 0 + mask
            # delta[:] = np.sqrt( np.abs(x * primal * diag) )
            
            # delta[:] = x
            # delta[:] = np.minimum(x * d, primal)

            old_zob2 = zob2
            
            zob = x - g[:, prev]
            zob = delta
            # delta = mask * zob
            
            zob2 = zob.dot( zob / d )
            beta = zob2 / old_zob2
            

            # delta[:] = zob

            # cond = r.dot(M.dot(r)) > r_prev.dot(M.dot(r_prev))
            # cond = zob.dot(zob) > f[:, prev].dot(f[:, prev])


            # cond = flag or delta.dot(d * delta) >= f[:, prev].dot(d * f[:, prev])
            cond = any( (x_prev > 0) != (x > 0) )
            # cond = beta > 1
            
            if reset and cond:
                g = np.zeros( (n, m) )
                f = np.zeros( (n, m) )
                k = np.zeros( (m, m) )
                flag = False
                skip = True
                print it, 'restart'
                
                
            g[:, index] = x
            f[:, index] = delta

            k[:, index] = f.transpose().dot( f[:, index] / d )
            k[index, :] = k[:, index].transpose()

            k[:, :] = f.transpose().dot(f)

            eps = 0 # 1e-8
            k[index, index] += eps
            rhs = np.copy( ones )
            rhs[index] += eps
            
            tmp[:] = np.linalg.lstsq(k, ones)[0]
            tmp /= sum(tmp)

            delta[:] = g.dot(tmp)

            if reset and not skip:
                _, a = max_step(x, delta)
                x[:] += a * (delta - x)
                skip = False
                # flag = any( (delta >= 0) != (x >= 0) )
                if a < 1: flag = True
            else:
                x[:] = delta
                

            # x[:] = delta
            
            index = (index + 1) % m
            it += 1
    res.__doc__ = "anderson{}({}) + {}".format('*' if 'metric' in kwargs else '',
                                               m,
                                               solver.__doc__)
    return res




def bokhoven(x, (M, q), **kwargs):
    '''bokhoven'''
    
    diag = np.diag(M)
    prec = kwargs.get('prec', diag)

    EpM = np.diag(prec) + M
    EpMinv = np.linalg.inv(EpM)

    n = q.size
    z = np.zeros(n)
    
    while True:
        zabs = np.abs(z)
        z[:] = -zabs + EpMinv.dot(2 * prec * zabs - q)
        x[:] = z + np.abs(z)
        yield


def bokhoven_gs(x, (M, q), **kwargs):
    '''bokhoven gs'''

    n = q.size
    diag = np.diag(M)
    
    prec = kwargs.get('prec', diag)
    EpM = np.diag(prec) + M    

    EpMinv = np.linalg.inv(EpM)
    
    z = np.zeros(n)
    zabs = np.abs(z)

    while True:

        for i in range(n):
            z[i] = -zabs[i] + EpMinv[i, :].dot(2 * prec * zabs - q)
            zabs[i] = abs(z[i])

        x[:] = z + zabs
        yield


def bokhoven_chol(x, (M, q), **kwargs):
    '''bokhoven chol'''

    n = q.size
    diag = np.diag(M)
    
    prec = kwargs.get('prec', diag)
    EpM = np.diag(prec) + M    

    L = np.linalg.cholesky(EpM)
    
    z = np.zeros(n)
    zabs = np.abs(z)

    Linv = np.linalg.inv(L)

    # u = Linv.dot(2.0 * prec * zabs - q)
    u = np.zeros(n)
    while True:

        # u = Linv.dot(2.0 * prec * zabs - q)
        for i in range(n):
            u[i] = ((2.0 * prec[i] * zabs[i] - q[i]) - L[i, :i].dot(u[:i])) / L[i, i]

        # print(np.linalg.norm(L.dot(u) - (2 * prec * zabs - q)))
            
        for j in range(n):
            i = n - 1 - j

            # u = Linv.dot(2 * prec * zabs - q)
            u[i] += ((2.0 * prec[i] * zabs[i] - q[i]) - L[i, :].dot(u)) / L[i, i]            
            
            # z[i] = Linv.T[i, :].dot(u) - zabs[i]
            # z[i] += (u[i] - L.T[i, :].dot(zabs + z)) / L[i, i]
            z[i] = -zabs[i] + (u[i] - L.T[i, i+1:].dot(z[i+1:] + zabs[i+1:])) / L[i, i]
            
            zabs[i] = abs(z[i])
            
        x[:] = z + zabs
        yield

