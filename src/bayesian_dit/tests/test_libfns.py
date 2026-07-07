import jax
import jax.numpy as jnp

def test_key_splitting():
    key = jax.random.key(8)
    B, N = 4, 5
    *key, new_keys = jax.random.split(key, (1, B, N))
    print(key.shape)
    print(new_keys)
    new_keys = jnp.asarray(new_keys)
    print(new_keys.shape)


if __name__=="__main__":
    test_key_splitting()