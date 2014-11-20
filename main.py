#!/usr/bin/env python

import krylov
import splitting

import lcp
import numpy as np
import math
import metric
import accel

def parse_args():
    """command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(description='benchmark LCP solvers')
    
    parser.add_argument('file', nargs='+', help='filename for LCP data')

    parser.add_argument('--iter', type=int, default=100,
                        help='iteration count')
    
    parser.add_argument('--fontsize', type=int, default=8,
                        help='font size in plot')
    
    parser.add_argument('--nolegend', action='store_true',
                        help='disable legend in plots')
    
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='precision')
    
    return parser.parse_args()



def pgs(x, (M, q)):
    """PGS"""
    return splitting.pgs(x, M, -q)

def pjacobi(x, (M, q)):
    """PJacobi"""
    return splitting.pjacobi(x, M, -q)


def solvers( (M, q) ):
    """solver list"""
    
    return [pgs,
            accel.nlnscg(pgs),
            # accel.nlnscg(pgs, metric = np.diag(M) ),
            accel.nlnscg(pgs, metric = np.diag(M), omega = 1.4 ),
        ]



args = parse_args()

# solve params
iterations = args.iter
precision = args.eps

# plot setup
cols = min(3, len(args.file) )
rows = int(math.ceil(len(args.file) / float(cols)))

import matplotlib
from matplotlib import pyplot as plt

for param in [ 'axes.titlesize',
               'axes.labelsize',
               'xtick.labelsize',
               'ytick.labelsize',
               'legend.fontsize' ]:
    matplotlib.rcParams[param] = args.fontsize

_, plots = plt.subplots(rows, cols)



for i, f in enumerate(args.file):
    
    (M, q) = lcp.load( f )
    
    # error metric 
    error = metric.minimum_norm( (M, q) )

    print 'file:', f
    print 'dim:', q.size
    print 'metric:', error.__doc__

    p = plots[ i / cols, i % cols ] if rows > 1 else plots[ i ] if cols > 1 else plots

    
    for s in solvers( (M, q) ):

        name = s.__doc__

        print '\t{}...'.format(name), 

        run = lcp.bench( (M, q), s,
                         iterations = iterations,
                         precision = precision,
                         metric = error )

        data = [e for k, e in run]
        
        print '\tit: {} \t eps: {:.2e}'.format(len(data), data[-1])

        # plot
        p.plot( data, label = name )
        p.set_title('{} (n = {})'.format(f, q.size))
        p.set_yscale('log')
        if not args.nolegend: p.legend()

        
plt.show()


