import jax
import jax.numpy as jnp
from flax import nnx
from simple_bayesian_vae.src.layers import BayesianConv2D

def eval_shape():
    my_convolve_layer = BayesianConv2D(
        1, 5, (3, 3), (2, 2), rngs=nnx.Rngs(jax.random.key(8)),
    )
    key = jax.random.key(0)
    my_input = jax.random.normal(key, (1, 45, 45, 1), dtype=jnp.float32)
    key_batch = jax.random.split(key, 1) # B=1

    out = jax.eval_shape(my_convolve_layer, my_input, key_batch)

    print(f"out_shape: {out.shape}\ndtype of out.shape: {jnp.asarray(out.shape)}")


if __name__=="__main__":
    eval_shape()