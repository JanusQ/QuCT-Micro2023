from jax import grad, jit, vmap, pmap
import numpy as np

X = np.array([[1],[2],[3]])
Y = np.array([[2,3],[3,4],[4,5]])
vmap(lambda params, y, x: print(params,x,y), in_axes=(None, 0, 0), out_axes=0)(None, X, Y)