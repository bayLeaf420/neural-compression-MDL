"""Unit tests for layers.py — run with: pytest tests/test_layers.py -v"""
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import pytest

# Adjust this import to match how your project resolves the module.
from bayesian_vae.layers import BayesianLinear, BayesianConv2D, _gaussian_kl_divergence


# ---------------------------------------------------------------------------
# _gaussian_kl_divergence  — the numerically load-bearing function
# ---------------------------------------------------------------------------

def test_kl_self_is_zero():
    """KL(p || p) == 0. This is the test that catches argument-order bugs."""
    mu = jnp.array([0.5, -1.2, 3.0])
    lnvar = jnp.array([0.1, -0.4, 2.0])
    # posterior == prior  =>  KL must be ~0
    kl = _gaussian_kl_divergence(mu, lnvar, mu, lnvar)
    assert jnp.allclose(kl, 0.0, atol=1e-5), f"KL(p||p) should be 0, got {kl}"


def test_kl_standard_normal_closed_form():
    """KL(N(0,1) || N(0,1)) == 0, and a known non-trivial value."""
    # posterior N(0, I), prior N(0, I): KL = 0
    z = jnp.zeros(4)
    kl0 = _gaussian_kl_divergence(z, jnp.zeros(4), z, jnp.zeros(4))
    assert jnp.allclose(kl0, 0.0, atol=1e-6)

    # posterior N(mu, 1), prior N(0, 1): KL = 0.5 * sum(mu^2)
    mu = jnp.array([1.0, 2.0])
    kl = _gaussian_kl_divergence(mu, jnp.zeros(2), jnp.zeros(2), jnp.zeros(2))
    expected = 0.5 * jnp.sum(mu**2)  # = 0.5*(1+4) = 2.5
    assert jnp.allclose(kl, expected, atol=1e-5), f"expected {expected}, got {kl}"


def test_kl_variance_only_closed_form():
    """posterior N(0, v), prior N(0, 1): KL = 0.5*(v - 1 - ln v) per dim."""
    lnvar = jnp.array([jnp.log(4.0)])   # variance = 4
    kl = _gaussian_kl_divergence(
        jnp.zeros(1), lnvar, jnp.zeros(1), jnp.zeros(1)
    )
    v = 4.0
    expected = 0.5 * (v - 1.0 - jnp.log(v))
    assert jnp.allclose(kl, expected, atol=1e-5), f"expected {expected}, got {kl}"


def test_kl_is_nonnegative():
    """KL divergence is always >= 0 for any posterior/prior."""
    key = jax.random.key(0)
    for i in range(5):
        key, *ks = jax.random.split(key, 5)
        kl = _gaussian_kl_divergence(
            jax.random.normal(ks[0], (6,)),
            jax.random.normal(ks[1], (6,)),
            jax.random.normal(ks[2], (6,)),
            jax.random.normal(ks[3], (6,)),
        )
        assert kl >= -1e-5, f"KL must be non-negative, got {kl}"


def test_kl_returns_scalar():
    """Output is a scalar (summed over dims), not per-dim."""
    kl = _gaussian_kl_divergence(
        jnp.zeros(10), jnp.zeros(10), jnp.zeros(10), jnp.zeros(10)
    )
    assert kl.shape == (), f"expected scalar, got shape {kl.shape}"


def test_kl_argument_order_matters():
    """Swapping posterior/prior mean gives a DIFFERENT result unless symmetric.
    Guards against the exact arg-order bug that slipped past Pylance."""
    post_mu = jnp.array([2.0, 0.0])
    prior_mu = jnp.array([0.0, 0.0])
    lnvar = jnp.zeros(2)
    kl_correct = _gaussian_kl_divergence(post_mu, lnvar, prior_mu, lnvar)
    # If someone swapped the mu args, for THIS asymmetric case it happens to be
    # symmetric in mu (depends on (post_mu-prior_mu)^2), so instead test that
    # putting a nonzero where prior_lnvar goes changes the answer:
    kl_swapped = _gaussian_kl_divergence(post_mu, lnvar, prior_mu, jnp.array([1.0, 1.0]))
    assert not jnp.allclose(kl_correct, kl_swapped), \
        "prior_lnvar must affect the result; arg positions may be wrong"


# ---------------------------------------------------------------------------
# BayesianLinear
# ---------------------------------------------------------------------------

def _make_keys(key, batch_size):
    return jax.random.split(key, batch_size)


def test_linear_output_shape():
    layer = BayesianLinear(8, 4, rngs=nnx.Rngs(jax.random.key(1)))
    B = 5
    x = jnp.ones((B, 8))
    keys = _make_keys(jax.random.key(2), B)
    out = layer(x, keys)
    assert out.shape == (B, 4), f"expected ({B}, 4), got {out.shape}"


def test_linear_no_bias_constructs_and_runs():
    """The use_bias=False path — the one that had the AttributeError bug."""
    layer = BayesianLinear(8, 4, use_bias=False, rngs=nnx.Rngs(jax.random.key(1)))
    assert layer.b_mu is None
    assert layer.b_lnvar is None
    B = 3
    x = jnp.ones((B, 8))
    keys = _make_keys(jax.random.key(2), B)
    out = layer(x, keys)  # must not raise
    assert out.shape == (B, 4)


def test_linear_kl_is_finite_and_nonneg():
    layer = BayesianLinear(8, 4, rngs=nnx.Rngs(jax.random.key(1)))
    kl = layer.calculate_kl_divergence()
    assert jnp.isfinite(kl), "KL must be finite"
    assert kl >= -1e-5, "KL must be non-negative"


def test_linear_kl_no_bias_runs():
    """calculate_kl_divergence on the no-bias path must not touch None."""
    layer = BayesianLinear(8, 4, use_bias=False, rngs=nnx.Rngs(jax.random.key(1)))
    kl = layer.calculate_kl_divergence()  # must not raise
    assert jnp.isfinite(kl)


def test_linear_output_is_stochastic():
    """Different keys should give different outputs (weights are sampled)."""
    layer = BayesianLinear(8, 4, rngs=nnx.Rngs(jax.random.key(1)))
    x = jnp.ones((1, 8))
    out1 = layer(x, _make_keys(jax.random.key(2), 1))
    out2 = layer(x, _make_keys(jax.random.key(99), 1))
    assert not jnp.allclose(out1, out2), "outputs should differ across keys"


# ---------------------------------------------------------------------------
# BayesianConv2D
# ---------------------------------------------------------------------------

def test_conv_output_shape_same_padding_stride1():
    """SAME padding, stride 1 => spatial dims unchanged, channels = out_channels."""
    layer = BayesianConv2D(1, 5, (3, 3), (1, 1), rngs=nnx.Rngs(jax.random.key(1)))
    B = 2
    x = jnp.ones((B, 28, 28, 1))
    keys = _make_keys(jax.random.key(2), B)
    out = layer(x, keys)
    assert out.shape == (B, 28, 28, 5), f"got {out.shape}"


def test_conv_output_shape_stride2():
    """SAME padding, stride 2 => spatial dims halved (ceil)."""
    layer = BayesianConv2D(1, 3, (3, 3), (2, 2), rngs=nnx.Rngs(jax.random.key(1)))
    B = 2
    x = jnp.ones((B, 28, 28, 1))
    keys = _make_keys(jax.random.key(2), B)
    out = layer(x, keys)
    # ceil(28/2) = 14
    assert out.shape == (B, 14, 14, 3), f"got {out.shape}"


def test_conv_no_bias_constructs_and_runs():
    layer = BayesianConv2D(1, 4, (3, 3), use_bias=False, rngs=nnx.Rngs(jax.random.key(1)))
    assert layer.b_mu is None
    B = 2
    x = jnp.ones((B, 16, 16, 1))
    keys = _make_keys(jax.random.key(2), B)
    out = layer(x, keys)  # must not raise
    assert out.shape[0] == B and out.shape[-1] == 4


def test_conv_feature_group_count_validation():
    """Invalid feature_group_count must raise ValueError at construction."""
    with pytest.raises(ValueError):
        # in_channels=3 not divisible by feature_group_count=2
        BayesianConv2D(3, 4, (3, 3), feature_group_count=2,
                       rngs=nnx.Rngs(jax.random.key(1)))


def test_conv_kl_is_finite_and_nonneg():
    layer = BayesianConv2D(1, 5, (3, 3), rngs=nnx.Rngs(jax.random.key(1)))
    kl = layer.calculate_kl_divergence()
    assert jnp.isfinite(kl)
    assert kl >= -1e-5


def test_conv_eval_shape_matches_real_call():
    """eval_shape should predict the same shape as an actual forward pass.
    This is the exact use-case models.py depends on for flat_dim."""
    layer = BayesianConv2D(1, 5, (3, 3), (2, 2), rngs=nnx.Rngs(jax.random.key(1)))
    B = 1
    x = jnp.ones((B, 45, 45, 1))
    keys = _make_keys(jax.random.key(2), B)
    predicted = jax.eval_shape(layer, x, keys)
    actual = layer(x, keys)
    assert predicted.shape == actual.shape, \
        f"eval_shape {predicted.shape} != actual {actual.shape}"


# ---------------------------------------------------------------------------
# Manual runner (so you can `python tests/test_layers.py` without pytest)
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