import jax
from jax import numpy as jnp
import numpy as np
import optax

def gen_data(x): 
    return 1/(12*x) + 1

def preidct(weights, x):
    return 1/(weights[0] * x) + weights[1]

X = jnp.arange(0, 100)
Y = gen_data(X)

weights = jax.random.normal(jax.random.PRNGKey(0), (2,))
opt = optax.adamw(learning_rate=1e-2)
opt_state = opt.init(weights)

for x, y in zip(X, Y):
    def loss(weights, x, y):
        return optax.l2_loss(preidct(weights, x) - y)
    
    grad_loss = jax.grad(loss)
    gradient = grad_loss(weights, x, y)
    updates, opt_state = opt.update(gradient, opt_state, weights)
    weights = optax.apply_updates(weights, updates)        

    loss_value = loss(weights, x, y)

grad_tanh = jax.grad(jax.numpy.tanh)

print(grad_tanh(0.2))


def func1(x): 
    return 1/x

grad_func1 = jax.grad(func1)
print(grad_func1(2.0))  # -1/x^2

def func2(x, y): 
    return 1/(x+y)

grad_func2 = jax.grad(func2, argnums=[0,1])
print(grad_func2(2.0, 1.0))



def func3(x, y): 
    return 1/(x[0] + x[1] + y)

grad_func3 = jax.grad(func3, argnums=[0,1])
print(grad_func3([2.0, 1.0], 1.0))