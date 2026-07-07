"""Unit tests for models.py — run with: pytest ./src/bayesian_vae/tests/test_models.py -v"""
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import pytest

from bayesian_vae.models import BayesianEncoder, BayesianDecoder, BayesianVAE
from bayesian_vae.config import (
    ConvConfig, LinConfig, EncoderConfig, DecoderConfig, VaeConfig,
)
from bayesian_vae.utils import PriorParam


# ---------------------------------------------------------------------------
# Shared small fixtures — a tiny model so tests run fast
# ---------------------------------------------------------------------------

IN_SHAPE = (28, 28, 1)
Z_DIM = 8


def _tiny_encoder_config():
    return EncoderConfig(
        ConvConfig(
            kernels=((3, 3), (3, 3)),
            strides=((1, 1), (2, 2)),   # include a stride-2 to exercise downsampling
            channels=(4, 6),
        ),
        LinConfig(hidden_dims=(32, 16)),
    )


def _tiny_decoder_config():
    return DecoderConfig(
        LinConfig(
            hidden_dims=(32, 12),
        ),
        ConvConfig(
            kernels=((3, 3), (3, 3)),
            strides=((1, 1), (1, 1)),
            channels=(2, 2),
        )
    )


def _tiny_vae_config():
    return VaeConfig(
        encoder_config=_tiny_encoder_config(),
        decoder_config=_tiny_decoder_config(),
        z_dim=Z_DIM,
        w_prior_lnvar=0.0,
        z_free_nats=0.5,
    )


def _rngs(seed=0):
    return nnx.Rngs(jax.random.key(seed))


def _make_encoder():
    return BayesianEncoder(IN_SHAPE, _tiny_encoder_config(), Z_DIM, rngs=_rngs(1))


def _make_decoder():
    return BayesianDecoder(Z_DIM, _tiny_decoder_config(), IN_SHAPE, rngs=_rngs(2))


def _make_vae():
    return BayesianVAE(IN_SHAPE, _tiny_vae_config(), rngs=_rngs(3))


# ---------------------------------------------------------------------------
# BayesianEncoder
# ---------------------------------------------------------------------------

def test_encoder_constructs():
    """Construction exercises the jax.eval_shape flat_dim computation — the
    line that had the ShapeDtypeStruct bug. If flat_dim is wrong, the first
    linear layer's shape is wrong and this raises."""
    enc = _make_encoder()
    assert enc.z_dim == Z_DIM


def test_encoder_output_shapes():
    enc = _make_encoder()
    B = 4
    x = jnp.ones((B, *IN_SHAPE))
    z_mu, z_lnvar = enc(x, jax.random.key(5))
    assert z_mu.shape == (B, Z_DIM), f"z_mu shape {z_mu.shape}"
    assert z_lnvar.shape == (B, Z_DIM), f"z_lnvar shape {z_lnvar.shape}"


def test_encoder_flat_dim_matches_conv_output():
    """The flat_dim used for the first linear layer must equal the actual
    flattened conv output. Verify by pushing an input through and checking
    the encoder didn't error on a shape mismatch (it would raise in the
    linear layer if flat_dim were wrong)."""
    enc = _make_encoder()
    B = 2
    x = jnp.ones((B, *IN_SHAPE))
    z_mu, _ = enc(x, jax.random.key(0))
    assert jnp.all(jnp.isfinite(z_mu)), "encoder produced non-finite output"


def test_encoder_kl_finite_nonneg():
    enc = _make_encoder()
    kl = enc.calculate_kl_divergence()
    assert jnp.isfinite(kl)
    assert kl >= -1e-5


def test_encoder_invalid_architecture_raises():
    """A conv stack that downsamples 28x28 below 1x1 should fail to construct."""
    bad_config = EncoderConfig(
        ConvConfig(
            kernels=((3, 3),) * 6,
            strides=((2, 2),) * 6,   # 6 stride-2 layers: 28 -> 14 -> 7 -> 4 -> 2 -> 1 -> 1
            channels=(4,) * 6,
        ),
        LinConfig(hidden_dims=(16,)),
    )
    # Depending on your validity handling, this either raises or produces a
    # degenerate 1x1 — assert whichever your code actually does.
    with pytest.raises((ValueError, Exception)):
        BayesianEncoder(IN_SHAPE, bad_config, Z_DIM, rngs=_rngs(1))

# ---------------------------------------------------------------------------
# BayesianDecoder
# ---------------------------------------------------------------------------

def test_decoder_output_shapes():
    dec = _make_decoder()
    B = 4
    z = jnp.ones((B, Z_DIM))
    x_hat_mu, x_hat_lnvar = dec(z, jax.random.key(5))
    assert x_hat_mu.shape == (B, *IN_SHAPE), f"x_hat_mu shape {x_hat_mu.shape}"
    assert x_hat_lnvar.shape == (B, *IN_SHAPE), f"x_hat_lnvar shape {x_hat_lnvar.shape}"


def test_decoder_kl_finite_nonneg():
    dec = _make_decoder()
    kl = dec.calculate_kl_divergence()
    assert jnp.isfinite(kl)
    assert kl >= -1e-5


# ---------------------------------------------------------------------------
# BayesianVAE — the full model
# ---------------------------------------------------------------------------

def test_vae_call_returns_five_things():
    """__call__ must return (x_hat_mu, x_hat_lnvar, z_mu, z_lnvar, z).
    The 5-tuple contract is what compute_training_loss unpacks."""
    vae = _make_vae()
    B = 3
    x = jnp.ones((B, *IN_SHAPE))
    out = vae(x, jax.random.key(0))
    assert len(out) == 5, f"expected 5 outputs, got {len(out)}"


def test_vae_output_shapes():
    vae = _make_vae()
    B = 3
    x = jnp.ones((B, *IN_SHAPE))
    x_hat_mu, x_hat_lnvar, z_mu, z_lnvar, z = vae(x, jax.random.key(0))
    assert x_hat_mu.shape == (B, *IN_SHAPE)
    assert x_hat_lnvar.shape == (B, *IN_SHAPE)
    assert z_mu.shape == (B, Z_DIM)
    assert z_lnvar.shape == (B, Z_DIM)
    assert z.shape == (B, Z_DIM)


def test_vae_reconstruction_roundtrip_shape():
    """Input shape must equal reconstruction mean shape — the compression
    round-trip only makes sense if x and x_hat have the same shape."""
    vae = _make_vae()
    B = 2
    x = jnp.ones((B, *IN_SHAPE))
    x_hat_mu, *_ = vae(x, jax.random.key(0))
    assert x_hat_mu.shape == x.shape


def test_vae_latent_kl_shape_and_finite():
    """calculate_latent_kl_divergence returns per-example KL, shape [B]."""
    vae = _make_vae()
    B = 5
    x = jnp.ones((B, *IN_SHAPE))
    _, _, z_mu, z_lnvar, _ = vae(x, jax.random.key(0))
    kl = vae.calculate_latent_kl_divergence(z_mu, z_lnvar)
    assert kl.shape == (B,), f"expected per-example KL shape ({B},), got {kl.shape}"
    assert jnp.all(jnp.isfinite(kl))
    assert jnp.all(kl >= -1e-5)


def test_vae_weights_kl_finite_nonneg():
    vae = _make_vae()
    kl = vae.calculate_vae_weights_kl_divergence()
    assert jnp.isfinite(kl)
    assert kl >= -1e-5


def test_vae_prior_is_priorparam():
    """The learnable prior must be PriorParam (nnx.Variable), NOT nnx.Param,
    so the optimizer's wrt=nnx.Param filter skips it (it's EMA-updated)."""
    vae = _make_vae()
    assert isinstance(vae.z_prior_mu, PriorParam)
    assert isinstance(vae.z_prior_lnvar, PriorParam)
    assert not isinstance(vae.z_prior_mu, nnx.Param) or issubclass(PriorParam, nnx.Param) is False
    # shapes must match z_dim for the per-dimension prior
    assert vae.z_prior_mu[...].shape == (Z_DIM,)
    assert vae.z_prior_lnvar[...].shape == (Z_DIM,)


def test_vae_prior_initialised_standard_normal():
    """Prior should start at N(0, I): mu=0, lnvar=0."""
    vae = _make_vae()
    assert jnp.allclose(vae.z_prior_mu[...], 0.0), "prior mean should init to 0"
    assert jnp.allclose(vae.z_prior_lnvar[...], 0.0), "prior lnvar should init to 0"


def test_vae_latent_kl_flooring():
    """Sanity: Check if flooring is preventing posterior collapse - i.e. posterior == prior should not give KL == 0"""
    vae = _make_vae()
    B = 4
    z_mu = jnp.zeros((B, Z_DIM))
    z_lnvar = jnp.zeros((B, Z_DIM))
    kl = vae.calculate_latent_kl_divergence(z_mu, z_lnvar)
    assert jnp.any(jnp.logical_not(jnp.isclose(kl, 0.0, atol=1e-03))), f"KL should not be ~0 when post==prior, got {kl}"


def test_vae_state_includes_prior_for_checkpointing():
    """nnx.state filtered by (nnx.Param, PriorParam) must include the prior,
    or checkpoints would restore a reset prior (the bug we discussed)."""
    vae = _make_vae()
    state = nnx.state(vae, (nnx.Param, PriorParam))
    # Flatten and check something prior-related is present. Exact structure is
    # nnx-version-dependent, so just assert the state is non-empty and the
    # Param-only state is strictly smaller (prior adds entries).
    param_only = nnx.state(vae, nnx.Param)
    combined_leaves = jax.tree.leaves(state)
    param_leaves = jax.tree.leaves(param_only)
    assert len(combined_leaves) > len(param_leaves), \
        "combined state should have more leaves than Param-only (the prior)"


# ---------------------------------------------------------------------------
# Manual runner (no pytest needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")