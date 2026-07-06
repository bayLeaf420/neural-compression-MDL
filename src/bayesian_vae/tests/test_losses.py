"""Unit tests for losses.py — run with: uv run pytest tests/test_losses.py -v"""
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import pytest

### RUN WITH: uv run pytest ./src/bayesian_vae/tests/test_losses.py -v ###

from bayesian_vae.losses import (
    erf_reconstruction_loss,
    compute_training_loss,
    compute_validation_reconstruction_loss,
    LossAux,
)
from bayesian_vae.models import BayesianVAE
from bayesian_vae.config import (
    ConvConfig, LinConfig, EncoderConfig, DecoderConfig, VaeConfig,
)


# ---------------------------------------------------------------------------
# Fixtures — a tiny VAE so tests run fast
# ---------------------------------------------------------------------------

IN_SHAPE = (28, 28, 1)
Z_DIM = 8


def _tiny_vae_config():
    return VaeConfig(
        encoder_config=EncoderConfig(
            ConvConfig(
                kernels=((3, 3), (3, 3)),
                strides=((1, 1), (2, 2)),
                channels=(4, 6),
            ),
            LinConfig(hidden_dims=(32, 16)),
        ),
        decoder_config=DecoderConfig(LinConfig(hidden_dims=(16, 32))),
        z_dim=Z_DIM,
        w_prior_lnvar=0.0,
        z_free_nats=0.5,
    )


@pytest.fixture
def vae():
    return BayesianVAE(IN_SHAPE, _tiny_vae_config(), rngs=nnx.Rngs(jax.random.key(3)))


@pytest.fixture
def batch():
    B = 4
    return jax.random.normal(jax.random.key(7), (B, *IN_SHAPE))


# ---------------------------------------------------------------------------
# erf_reconstruction_loss — the numerically load-bearing function
# ---------------------------------------------------------------------------

def test_recon_loss_is_nonnegative():
    """Discretized NLL is -log(probability mass), and mass <= 1, so NLL >= 0 always.
    This is THE key property that fixed the negative-loss pathology."""
    key = jax.random.key(0)
    shape = (2, *IN_SHAPE)
    mean = jax.random.uniform(key, shape)          # in [0,1], valid pixel range
    lnvar = jax.random.normal(jax.random.split(key)[0], shape)
    target = jax.random.uniform(jax.random.split(key)[1], shape)
    loss = erf_reconstruction_loss(mean, lnvar, target)
    assert loss >= 0.0, f"discretized NLL must be non-negative, got {loss}"


def test_recon_loss_confident_correct_is_low():
    """Perfect prediction (mean==target) with SMALL variance concentrates mass
    in the correct bin -> NLL near 0. Small variance so the bin captures most mass."""
    shape = (1, 4, 4, 1)
    target = jnp.full(shape, 0.5)                  # mid-range pixel
    mean = jnp.full(shape, 0.5)                    # perfect prediction
    lnvar = jnp.full(shape, -14.0)                 # tiny variance: std ~ 9e-4 ~ bin width
    loss = erf_reconstruction_loss(mean, lnvar, target)
    # With std ~ bin width and perfect centering, most mass is in the bin,
    # so NLL should be small (near 0, well under 1 per pixel).
    assert loss < 1.0, f"confident correct prediction should give low NLL, got {loss}"


def test_recon_loss_confident_wrong_is_high():
    """Confident (small variance) but WRONG prediction -> almost no mass in the
    target's bin -> large NLL. This is the behaviour that penalises overconfidence."""
    shape = (1, 4, 4, 1)
    target = jnp.full(shape, 0.9)
    mean = jnp.full(shape, 0.1)                    # confidently predicts wrong value
    lnvar = jnp.full(shape, -14.0)                 # tiny variance
    loss = erf_reconstruction_loss(mean, lnvar, target)
    assert loss > 10.0, f"confident wrong prediction should give high NLL, got {loss}"


def test_recon_loss_wrong_worse_than_right():
    """Monotonicity: at fixed variance, a worse prediction gives higher NLL."""
    shape = (1, 4, 4, 1)
    target = jnp.full(shape, 0.5)
    lnvar = jnp.zeros(shape)
    close = erf_reconstruction_loss(jnp.full(shape, 0.5), lnvar, target)   # perfect
    far = erf_reconstruction_loss(jnp.full(shape, 0.1), lnvar, target)     # off
    assert far > close, "worse prediction should give higher NLL"


def test_recon_loss_is_finite():
    shape = (2, *IN_SHAPE)
    key = jax.random.key(3)
    mean = jax.random.uniform(key, shape)
    lnvar = jnp.zeros(shape)
    target = jax.random.uniform(jax.random.split(key)[1], shape)
    assert jnp.isfinite(erf_reconstruction_loss(mean, lnvar, target))


# ---------------------------------------------------------------------------
# compute_training_loss — composition and the tuple contract
# ---------------------------------------------------------------------------

def test_training_loss_return_structure(vae, batch):
    """Must return (total_loss, LossAux, PostLog) — the 3-tuple that the
    loss_fn wrapper bundles for has_aux. Wrong arity here caused the
    'too many values to unpack' error."""
    out = compute_training_loss(vae, batch, jax.random.key(0), jnp.asarray(0.01))
    assert len(out) == 3, f"expected 3 return values, got {len(out)}"
    total_loss, aux, post_log = out
    assert isinstance(aux, LossAux)
    # post_log carries the latent stats for the EMA prior update
    assert hasattr(post_log, "z_mu")
    assert hasattr(post_log, "z_lnvar")


def test_training_loss_is_scalar_and_finite(vae, batch):
    total_loss, aux, _ = compute_training_loss(
        vae, batch, jax.random.key(0), jnp.asarray(0.01)
    )
    assert total_loss.shape == (), "total loss must be scalar"
    assert jnp.isfinite(total_loss)


def test_training_loss_components_finite(vae, batch):
    _, aux, _ = compute_training_loss(vae, batch, jax.random.key(0), jnp.asarray(0.01))
    assert jnp.isfinite(aux.reconstruction_loss)
    assert jnp.isfinite(aux.latent_kl_divergence)
    assert jnp.isfinite(aux.weight_kl_divergence)


def test_training_loss_is_sum_of_components(vae, batch):
    """total_loss should equal recon + latent_kl + weight_kl (the composition)."""
    total_loss, aux, _ = compute_training_loss(
        vae, batch, jax.random.key(0), jnp.asarray(0.01)
    )
    expected = (
        aux.reconstruction_loss
        + aux.latent_kl_divergence
        + aux.weight_kl_divergence
    )
    assert jnp.allclose(total_loss, expected, atol=1e-4), \
        f"total {total_loss} != sum of parts {expected}"


def test_training_loss_kl_components_nonneg(vae, batch):
    """KL divergences are non-negative; reconstruction NLL can be any sign."""
    _, aux, _ = compute_training_loss(vae, batch, jax.random.key(0), jnp.asarray(0.01))
    assert aux.latent_kl_divergence >= -1e-4
    assert aux.weight_kl_divergence >= -1e-4


def test_training_loss_kl_weight_scale_effect(vae, batch):
    """Larger kl_weight_scale should increase the weight-KL contribution,
    hence (all else equal) the total loss, since weight_kl scales with it."""
    key = jax.random.key(0)
    loss_small, aux_small, _ = compute_training_loss(vae, batch, key, jnp.asarray(0.0))
    loss_large, aux_large, _ = compute_training_loss(vae, batch, key, jnp.asarray(1.0))
    # With scale=0 the weight-KL term is zero; with scale=1 it's positive.
    assert aux_small.weight_kl_divergence == pytest.approx(0.0, abs=1e-5)
    assert aux_large.weight_kl_divergence > 0.0


# ---------------------------------------------------------------------------
# compute_validation_reconstruction_loss
# ---------------------------------------------------------------------------

def test_validation_loss_is_scalar_and_finite(vae, batch):
    loss = compute_validation_reconstruction_loss(vae, batch, jax.random.key(0))
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_validation_loss_is_pure_reconstruction(vae, batch):
    """Validation loss should be reconstruction-only (no KL terms), so it
    should match the recon component of training loss for the same key."""
    key = jax.random.key(0)
    val_loss = compute_validation_reconstruction_loss(vae, batch, key)
    # Training loss uses the same model(x, key) call internally; the recon
    # component should match the validation reconstruction (same key => same
    # sampled weights/latents => same x_hat).
    _, aux, _ = compute_training_loss(vae, batch, key, jnp.asarray(0.01))
    assert jnp.allclose(val_loss, aux.reconstruction_loss, atol=1e-4), \
        f"val {val_loss} != training recon {aux.reconstruction_loss}"


# ---------------------------------------------------------------------------
# Manual runner (no pytest) — note: fixtures won't work here, so this only
# runs the non-fixture tests. Use pytest for the full suite.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Run with: uv run pytest bayesian_vae/tests/test_losses.py -v")
    print("(This file uses pytest fixtures, so the manual runner is limited.)")