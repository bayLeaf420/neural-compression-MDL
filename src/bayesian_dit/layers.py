import jax 
import jax.numpy as jnp
from flax import nnx 

from bayesian_vae.layers import BayesianLinear


def apply_rotary_emb(
        xq: jax.Array, 
        xk: jax.Array,
        freqs_cis: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    freqs_cis = freqs_cis[jnp.newaxis, :, jnp.newaxis, :]
    xq_ = jax.lax.bitcast_convert_type(xq.reshape(*xq.shape[:-1], -1, 2), jnp.complex64)
    xk_ = jax.lax.bitcast_convert_type(xk.reshape(*xk.shape[:-1], -1, 2), jnp.complex64)
    xq_out = jax.lax.bitcast_convert_type(xq_ * freqs_cis, jnp.float32).
    


class RMSNorm(nnx.Module):
    """Defines a normalisation layer"""
    def __init__(self, hid_size, eps=1e-6):
        super().__init__()
        self.weight = nnx.Param(jnp.ones((hid_size,)))
        self.var_eps = eps

    def __call__(self, hid_states):
        """Calculate RMSNorm.

        Args:
            hid_states: [B, N, C], input
        """
        var = jnp.mean(hid_states ** 2, -1, keepdims=True) # [B, N, C]
        hid_states = hid_states * jax.lax.rsqrt(var + self.var_eps)
        return self.weight * hid_states # [B, N, 1] * [B, N, C] -> broadcasting


class BayesRotaryAttn(nnx.Module):
    """Bayesian Rotary attention. Using old Bayesian Linear layer for linal.

    Args:
        dim: Attention dimensions. 
        num_heads: Number of attention heads
        
    """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            use_qkv_bias: bool = False,
            use_qk_norm: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            *,
            rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, (
            "dim should be divisible by num heads."
            f"Currently: dim={dim}, num_heads={num_heads}"
        )
        self.dim = dim
        self.num_heads = num_heads 
        self.head_dim = dim // num_heads 
        self.scale = self.head_dim ** -0.5

        # Bayesian linear expects [B, d_in]. v-map over 1 because input is of
        # shape [B, N, d_in] where d_in == dim. Keys will be passed as [B, N]
        self.qkv = jax.vmap(
            BayesianLinear(dim, dim * 3, use_bias=use_qkv_bias, rngs=rngs),
            in_axes=(1, 1)
        )
        self.q_norm = RMSNorm(self.head_dim) if use_qk_norm else jax.nn.identity
        self.k_norm = RMSNorm(self.head_dim) if use_qk_norm else jax.nn.identity
        self.attn_drop = nnx.Dropout(attn_drop, rngs=rngs)
        self.proj = BayesianLinear(dim, dim, rngs=rngs)
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)

    def __call__(self, x: jax.Array, pos: jax.Array, mask: jax.Array, key: jax.Array) -> jax.Array:
        B, N, C = x.shape 

        qkv_keys = jax.random.split(key, (B, N))
        qkv_keys = jnp.asarray(qkv_keys)
        qkv = self.qkv(x, qkv_keys).reshape(B, N, 3, C)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freq_cis=pos)
