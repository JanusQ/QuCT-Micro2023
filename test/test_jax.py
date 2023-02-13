import random
import matplotlib.pyplot as plt
from tensorcircuit.backends.jax_backend import optax_optimizer

from sklearn.utils import shuffle

# import tensorflow as tf
import numpy as np
# 还得传参数进去
from jax import numpy as jnp
import jax
from jax import grad, jit, vmap, pmap

# var1 = tf.Variable(10.0)

# t  = jax.random.normal(shape=[8, 32], key=jax.random.PRNGKey(0)),

def learning_function(params, x)-> jnp.ndarray:
    # return (params[0]*a + params[1]*b + params[2]*c + params[3]*d)  * (params[4]*a + params[5]*b + params[6]*c + params[7]*d) # * (params[8]*b + params[9]*c)
    # return (params[0]*a + params[1]*b + params[2]*c + params[3]*d)
    r  = jnp.dot(x, params[0]) * jnp.dot(x, params[1]) * jnp.dot(x, params[2]) 
    return r

def target_function(x):
    return learning_function(np.arange(1,13).reshape(3,4,1), x)
    # return  (2*a + b + 3*c + 4*d) * (5*a + 6*b + 7*c + 9*d) * (3*b + 7*c)
    # return  (2*a + b + 3*c + 4*d)

# jnp.abs
# @jax.jit
def loss(x, parameters, y):
    return (y - learning_function(parameters, x))**2

# jnp.abs
# optax.l2_loss
# jax.debug.print.
@jax.jit
def batch_loss(parameters, X, Y):
    errors = vmap(loss, in_axes=(0, None, 0), out_axes=0)(X, parameters, Y) # in_axes应该是传几个参数就扫几个
    # error = jnp.array([(y - learning_function(parameters, x))**2  for x,y in zip(X,Y)])
    # print(errors)
    # print(error.mean())
    # jax.debug.print(jnp.where(jnp.any(parameters<0), 0, jnp.max(-parameters)*1e5))
    # parameters = parameters.reshape((1,12))
    return errors.mean() #- np.mean(parameters)
    # + jnp.where(jnp.any(parameters<0), 0, jnp.max(-parameters)*1e5)
    # parameters.reshape((1,12))
    # (y - learning_function(parameters, x))**2

# params = jnp.zeros([3,4,1], dtype=np.float32)
params = jnp.array([15]*12, dtype=np.float32).reshape(3,4,1) + jax.random.normal(shape=[3,4,1], key=jax.random.PRNGKey(0))
# params = jnp.arange(1,13, dtype=np.float32).reshape(3,4,1)  + jax.random.normal(shape=[3,4,1], key=jax.random.PRNGKey(0))
# params = np.arange(1,13).reshape(3,4,1)
# params = jax.random.normal(shape=[3,4,1], key=jax.random.PRNGKey(0))
# params = jnp.arange(1,13, dtype=np.float32).reshape(3,4,1)
# parameters.reshape((1,12))
import optax
optimizer = optax.adam(10e-2)
# optimizer = optax.adamw(10e-2)
opt_state = optimizer.init(params)

# print(learning_function(1,2,3,4 ,params))

batch_size = 100
X = np.array([np.random.randint(0, 10, size=(1,4)) for i in range(10000)])
Y = np.array([target_function(x) for x in X])


# 现在看来batch是很重要的
best_loss = 1e30
best_parmas = None
for epoch in range(100):
    # x = [random.random() * 10, random.random() * 10, random.random() * 10, random.random() * 10]
    # x = jax.random.normal(shape=(4,1), key=jax.random.PRNGKey(0))
    # y = 
    # print(x, y)
    # jax.random.shuffle()
    X,Y = shuffle(X, Y)
    for start in range(0, len(X), batch_size):
        loss_value, gradient = jax.value_and_grad(batch_loss)(params, X[start: start+batch_size], Y[start: start+batch_size])
        updates, opt_state = optimizer.update(gradient, opt_state, params)
        params = optax.apply_updates(params, updates)

        # 这个是可行的
        params = params.at[params > 13].set(13)
        params = params.at[params < 0].set(0)


    # if loss_value < best_loss:
    #     best_parmas = params
    #     best_loss = loss_value

    # if _%100 == 0:
    # print(epoch ,loss_value)  #, params
    #     # print(best_loss, best_parmas)
    # if epoch%10 == 0:
    #     print(params)
# 106566610000
# 31427135000
# 5229225
# 49173