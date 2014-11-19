#!/usr/bin/env python

import krylov
import splitting

import lcp
import numpy as np
import math
import metric
import accel

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='benchmark LCP solvers.')
    parser.add_argument('file', help='filename for LCP data')
    parser.add_argument('--iter', type=int, default=100,
                        help='iteration count')

    parser.add_argument('--eps', type=float, default=1e-8,
                   help='precision')

    return parser.parse_args()



# some adapters
def pgs(x, (M, q)):
    """PGS"""
    return splitting.pgs(x, M, -q)

def pjacobi(x, (M, q)):
    """PJacobi"""
    return splitting.pjacobi(x, M, -q)


args = parse_args()

(M, q) = lcp.load( args.file )
iterations = args.iter
precision = args.eps


# solver list
solvers = [pgs, pjacobi,
           accel.nlnscg(pgs), accel.nlnscg(pjacobi) ]

# error metric 
error = metric.lcp_merit( (M, q) )


from matplotlib import pyplot as plt

print 'file:', args.file

for s in solvers:

    name = s.__doc__

    run = lcp.bench( (M, q), s,
                     iterations = iterations,
                     precision = precision,
                     metric = error )
    
    print 'running {}...'.format(name), 
    data = [e for k, e in run]
    print 'ok'
    
    plt.plot( data, label = name )


plt.yscale('log')
plt.legend()
plt.title( error.__doc__.title() )
plt.show()
    
                          
