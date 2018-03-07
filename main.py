#!/usr/bin/env python

import krylov
import splitting

import tool
import numpy as np
import math
import metric
import accel

def parse_args():
    """command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(description='benchmark LCP solvers')
    
    parser.add_argument('filenames', nargs='+', help='filename for LCP data')

    parser.add_argument('-n', '--iter', type=int, default=100,
                        help='iteration count')
    
    parser.add_argument('--fontsize', type=int, default=8,
                        help='font size in plot')
    
    parser.add_argument('--legend', type = int, default = 1,
                        help='enable/disable legend in plots')

    parser.add_argument('--ms', type = int, default = 0,
                        help='use mass-splitting when available')
    
    
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='precision')

    parser.add_argument('-i', '--interactive', action = 'store_true',
                        help='drop to a python shell once finished')

    
    return parser.parse_args()



def pgs(x, (M, q), **kwargs):
    """pgs"""
    return splitting.pgs(x, M, -q, **kwargs)

def pjacobi(x, (M, q), **kwargs):
    """pjacobi"""
    return splitting.pjacobi(x, M, -q, **kwargs)


def solvers( (M, q), **kwargs ):
    """solver list"""

    n = q.size
    ms = kwargs.get('ms', None)

    s = pgs
    s = pjacobi
    
    def mass_splitting(x, (M, q), **kwargs):
        """mass-splitting"""
        return s(x, (M, q), diag = ms, omega = kwargs.get('omega', 1.0))

    mass_splitting.__doc__ += ' ({})'.format(s.__doc__)
    
    base = s if ms is None else mass_splitting


    n = q.size
    opts = { 'diag': ms, 'omega': kwargs.get('omega', 1.0)} if ms is not None else {
        'diag': np.diag(M), 'omega': 2.0 / n
    }


    def wrap(s, **kwargs):
        def res(x, lcp):
            return s(x, lcp, **kwargs)
        res.__doc__ = s.__doc__
        return res
    
    
    return [ # base,
             # accel.nlnscg(base, diag = opts['diag']),
             # accel.nlnscg_ls(base, **opts),
             # wrap(accel.cg_pjacobi, **opts),
             # wrap(accel.cr_pjacobi, **opts),
             # wrap(accel.doubidou, **opts),
             # accel.anderson(base, m = 4, reset = False), #  metric = opts['diag'] ),
             # wrap(accel.andy, **opts),
             pgs,
             wrap(accel.bokhoven),
             wrap(accel.bokhoven_gs),
             wrap(accel.bokhoven_chol),        
             ] 




args = parse_args()

# solve params
iterations = args.iter
precision = args.eps

# plot setup
cols = min(3, len(args.filenames) )
rows = int(math.ceil(len(args.filenames) / float(cols)))

import matplotlib
from matplotlib import pyplot as plt

for param in [ 'axes.titlesize',
               'axes.labelsize',
               'xtick.labelsize',
               'ytick.labelsize',
               'legend.fontsize' ]:
    matplotlib.rcParams[param] = args.fontsize

# _, plots = plt.subplots(rows, cols)
_, plots = plt.subplots()

def bench(filename, **kwargs):
    
        (M, q) = tool.load_lcp( filename )

        print 'symmetry check:', np.linalg.norm(M - M.transpose())

        if args.ms:
            try:
                ext = '.ms'
                
                split = f.split('.')
                if split[-1] == 'gz':
                    f = '.'.join(split[:-1])
                    ext += '.gz'
                    
                kwargs['ms'] = tool.load_vec( f + ext )
            except:
                pass

        # error metric 
        error = metric.lcp_error( (M, q),
                                  primal = True,
                                  dual = False,
                                  compl = True )

        error = metric.minimum_norm( (M, q) ) #, metric = np.diag(M) )


        print 'file:', filename
        print 'dim:', q.size
        print 'metric:', error.__doc__

        p = plots #  i / cols, i % cols ] if rows > 1 else plots[ i ] if cols > 1 else plots

        np.random.seed()
        # initial = np.random.rand( q.size )
        initial = np.zeros(q.size)
        
        p.cla()
        for s in solvers( (M, q), **kwargs ):

            name = s.__doc__

            print '\t{}...'.format(name), 

            run = tool.bench( (M, q), s,
                              iterations = iterations,
                              precision = precision,
                              metric = error,
                              initial = initial)

            data = [e for k, e in run]

            print '\tit: {} \t eps: {:.2e}'.format(len(data), data[-1])

            # plot
            p.plot( data, label = name )
            p.set_title('{} (n = {})'.format(filename, q.size))
            p.set_yscale('log')
            if args.legend: p.legend()

        plt.draw()



plt.ion()

for filename in args.filenames:
    bench(filename)
    plt.show()
    print('press enter to continue...')
    raw_input()
    
if args.interactive:
    import code
    code.interact(None, None, locals())

