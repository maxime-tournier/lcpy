Benchmark LCP solvers on difficult contact problems. Requires Python
>= 2.7, NumPy and matplotlib.

Usage
-----

```
$ python main.py -h
usage: main.py [-h] [--iter ITER] [--eps EPS] file

benchmark LCP solvers.

positional arguments:
  file         filename for LCP data

optional arguments:
  -h, --help   show this help message and exit
  --iter ITER  iteration count
  --eps EPS    precision
```



Solvers
-------

- (Projected) Gauss-Seidel
- (Projected) Jacobi
- Non-Smooth Non-Linear Conjugate Gradient acceleration

- TODO Mass-Splitting Jacobi
- TODO many more :D

Metrics
-------

- LCP primal+dual+complementarity errors
- Convex merit function norm( min(x, M x + q) )

Data Format
-----------

Data files are plaintext files, with:
- first line: dimension (1 integer)
- n next lines: matrix (n lines of n floats)
- last line: vector (1 line of n floats)




