import jax
import jax.numpy as jnp
from flax import nnx
from bayesian_vae.layers import BayesianConv2D

def eval_shape():
    my_convolve_layer = BayesianConv2D(
        1, 5, (3, 3), (16, 16), rngs=nnx.Rngs(jax.random.key(8)),
    )
    key = jax.random.key(0)
    my_input = jax.random.normal(key, (1, 20, 20, 1), dtype=jnp.float32)
    key_batch = jax.random.split(key, 1) # B=1

    out = jax.eval_shape(my_convolve_layer, my_input, key_batch)

    print(f"out_shape: {out.shape}\ndtype of out.shape: {jnp.asarray(out.shape)}")

def test_stateful_comp():
    class Counter:
        def __init__(self):
            self.counter = jnp.asarray(0)
        @jax.jit
        def count(self):
            self.counter += 1
            return self.counter
        def reset(self):
            self.counter = 0
    
    counter = Counter()
    for _ in range(3):
        print(counter.count())

if __name__=="__main__":
    # eval_shape()
    test_stateful_comp()