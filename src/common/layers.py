import jax 
import jax.numpy as jnp
from flax import nnx
import einops

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    Return sincos embeddings over whole image. 
    """
    grid_h = jnp.arange(grid_size, dtype=jnp.float32) # (H)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32) # (W,)
    grid = jnp.meshgrid(grid_w, grid_h)  
    grid = jnp.stack(grid, axis=0) # Returns (2, H, W) shaped array

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Use 1D sincos embedding to sin-cos embedding for 2D grid. Returns a unique
    sin-cos embedding for each position in the grid. 
    """
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2) returned
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Args:
        embed_dim (int): output dimension for each position
        pos (jax.Array): a list of positions to be encoded: size (M,)
    Returns:
        jax.Array: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def apply_adaln(x, shift, scale):
    """Calculate AdaLN"""
    return x * (1 + scale) + shift
    

def precompute_freqs_cis_2d(dim, height, width, theta = 10000.0, scale=1.0):
    """Return precomputed rotary embeddings to add to query/key"""
    if isinstance(scale, float):
        scale = (scale, scale)
    x_pos = jnp.linspace(0, width * scale[0], width) # type: ignore
    y_pos = jnp.linspace(0, height * scale[1], height) # type: ignore
    y_pos, x_pos = jnp.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 4, dtype=jnp.float32)[: (dim // 4)] / dim)) # Hc/4
    x_freqs = x_pos[:, jnp.newaxis] * freqs # N Hc/4
    y_freqs = y_pos[:, jnp.newaxis] * freqs # N Hc/4
    x_cis = jnp.exp(1j * x_freqs)
    y_cis = jnp.exp(1j * y_freqs) # JAX has complex support
    freqs_cis = jnp.stack([x_cis, y_cis], axis=-1)
    freqs_cis = freqs_cis.reshape(height*width, -1)
    return freqs_cis


def apply_rotary_emb(x, freqs_cis):
    """Apply RoPE embedding to an input vector"""

    freqs_cis = freqs_cis[None, :, None, :]   # [1, N, 1, dim//2]
    init_shape = x.shape

    xc = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 2)
    xc = jax.lax.complex(x[..., 0], x[..., 1])   # view_as_complex

    xc = xc * freqs_cis

    xr = jnp.stack([jnp.real(xc), jnp.imag(xc)], axis=-1)  # view_as_real
    x_out = xr.reshape(init_shape)

    return x_out.astype(x)


class RMSNorm(nnx.Module):
    def __init__(self, dim, eps=1e-6, *, rngs=None):
        self.weight = nnx.Param(jnp.ones(dim))
        self.eps = eps

    def __call__(self, x):
        in_dtype = x.dtype
        x = x.astype(jnp.float32)
        var = jnp.mean(x ** 2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(var + self.eps)
        return self.weight * x.astype(in_dtype)


class RotaryAttention(nnx.Module):
    """Rotary Attention Class"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=True,
                 attn_drop=0.0, proj_drop=0.0, norm_layer=RMSNorm, *, rngs):
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
        self.q_norm = norm_layer(self.head_dim, rngs=rngs) if qk_norm else (lambda x: x)
        self.k_norm = norm_layer(self.head_dim, rngs=rngs) if qk_norm else (lambda x: x)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)

    def __call__(self, x, pos, mask=None):
        """MHA implementation

        
        """
        # One projection, split into q/k/v with named axes — no permute.
        # qkv output packs channels as [3, H, hd] (matches the original reshape).
        q, k, v = einops.rearrange(
            self.qkv(x), "b n (three h d) -> three b n h d",
            three=3, h=self.num_heads,
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cis=pos)   # operates in [B, N, H, hd], stays put
        k = apply_rotary_emb(k, freqs_cis=pos)

        # Fold the 1/sqrt(hd) scale into q (small tensor) instead of the [B,H,N,N] scores.
        q = q * self.scale

        # Attention with head as a batched named axis — no transpose to [B, H, N, hd].
        attn = jnp.einsum("bqhd,bkhd->bhqk", q, k)
        if mask is not None:
            attn = attn + mask
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v)

        out = einops.rearrange(out, "b q h d -> b q (h d)")
        out = self.proj(out)
        out = self.proj_drop(out)
        return out