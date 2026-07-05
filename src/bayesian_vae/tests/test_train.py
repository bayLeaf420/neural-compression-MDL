"""Unit tests for train.py — run with: uv run pytest tests/test_train.py -v"""
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import pytest

from bayesian_vae.train import (
    prior_train_step_ema,
    train_step,
    validation_step,
    build_model,
)
from bayesian_vae.models import BayesianVAE
from bayesian_vae.utils import PriorParam


IN_SHAPE = (28, 28, 1)
DECAY = 0.99   # a test decay value for train_step's static arg


@pytest.fixture
def model():
    return build_model(jax.random.key(0))


@pytest.fixture
def optimizer(model):
    return nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)


@pytest.fixture
def batch():
    return jax.random.normal(jax.random.key(1), (4, *IN_SHAPE))


# ---------------------------------------------------------------------------
# prior_train_step_ema — pure function
# ---------------------------------------------------------------------------

def test_ema_returns_two_arrays():
    z_dim = 5
    new_mu, new_lnvar = prior_train_step_ema(
        jnp.zeros(z_dim), jnp.zeros(z_dim),
        jnp.ones((8, z_dim)), jnp.zeros((8, z_dim)), 0.9,
    )
    assert new_mu.shape == (z_dim,)
    assert new_lnvar.shape == (z_dim,)


def test_ema_decay_one_keeps_prior_unchanged():
    z_dim = 4
    prior_mu = jnp.array([1.0, 2.0, 3.0, 4.0])
    prior_lnvar = jnp.array([0.5, 0.5, 0.5, 0.5])
    new_mu, new_lnvar = prior_train_step_ema(
        prior_mu, prior_lnvar,
        jnp.ones((6, z_dim)) * 99.0, jnp.ones((6, z_dim)) * 99.0, 1.0,
    )
    assert jnp.allclose(new_mu, prior_mu), "decay=1 must leave prior mean unchanged"
    # Note: lnvar won't be EXACTLY unchanged because of the 1e-6 floor/shift in
    # the log; allow tolerance rather than exact equality.
    assert jnp.allclose(new_lnvar, prior_lnvar, atol=1e-4), \
        "decay=1 should leave prior lnvar ~unchanged"


def test_ema_decay_zero_jumps_to_batch_mean():
    z_dim = 3
    z_mu = jnp.tile(jnp.array([1.0, 2.0, 3.0]), (10, 1))
    new_mu, _ = prior_train_step_ema(
        jnp.zeros(z_dim), jnp.zeros(z_dim), z_mu, jnp.zeros((10, z_dim)), 0.0,
    )
    assert jnp.allclose(new_mu, jnp.array([1.0, 2.0, 3.0]), atol=1e-5)


def test_ema_variance_stays_positive():
    z_dim = 5
    key = jax.random.key(2)
    k = jax.random.split(key, 4)
    new_mu, new_lnvar = prior_train_step_ema(
        jax.random.normal(k[0], (z_dim,)),
        jax.random.normal(k[1], (z_dim,)),
        jax.random.normal(k[2], (8, z_dim)),
        jax.random.normal(k[3], (8, z_dim)),
        0.99,
    )
    assert jnp.all(jnp.isfinite(new_mu))
    assert jnp.all(jnp.isfinite(new_lnvar)), "lnvar finite => variance stayed > 0"


def test_ema_moves_prior_partway():
    z_dim = 2
    z_mu = jnp.tile(jnp.array([10.0, 10.0]), (5, 1))
    new_mu, _ = prior_train_step_ema(
        jnp.zeros(z_dim), jnp.zeros(z_dim), z_mu, jnp.zeros((5, z_dim)), 0.9,
    )
    assert jnp.all(new_mu > 0.0) and jnp.all(new_mu < 10.0)


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

def test_build_model_returns_vae():
    assert isinstance(build_model(jax.random.key(0)), BayesianVAE)


def test_build_model_deterministic():
    m1 = build_model(jax.random.key(42))
    m2 = build_model(jax.random.key(42))
    l1 = jax.tree.leaves(nnx.state(m1, nnx.Param))
    l2 = jax.tree.leaves(nnx.state(m2, nnx.Param))
    assert all(jnp.allclose(a, b) for a, b in zip(l1, l2))


def test_build_model_prior_is_priorparam():
    m = build_model(jax.random.key(0))
    assert isinstance(m.z_prior_mu, PriorParam)


# ---------------------------------------------------------------------------
# train_step — behavioural (note the 6th arg: decay)
# ---------------------------------------------------------------------------

def test_train_step_runs_and_returns(model, optimizer, batch):
    loss, aux = train_step(
        model, optimizer, batch, jax.random.key(0), jnp.asarray(0.01), DECAY
    )
    assert jnp.isfinite(loss)
    assert jnp.isfinite(aux.reconstruction_loss)


def test_train_step_updates_params(model, optimizer, batch):
    before = [jnp.array(x) for x in jax.tree.leaves(nnx.state(model, nnx.Param))]
    train_step(model, optimizer, batch, jax.random.key(0), jnp.asarray(0.01), DECAY)
    after = jax.tree.leaves(nnx.state(model, nnx.Param))
    assert any(not jnp.allclose(b, a) for b, a in zip(before, after)), \
        "training step should update at least some parameters"


def test_train_step_prior_stays_finite(model, optimizer, batch):
    train_step(model, optimizer, batch, jax.random.key(0), jnp.asarray(0.01), DECAY)
    assert jnp.all(jnp.isfinite(model.z_prior_mu[...]))
    assert jnp.all(jnp.isfinite(model.z_prior_lnvar[...]))


# ---------------------------------------------------------------------------
# validation_step
# ---------------------------------------------------------------------------

def test_validation_step_runs(model, batch):
    loss = validation_step(model, batch, jax.random.key(0))
    assert jnp.isfinite(loss)
    assert loss.shape == ()


def test_validation_step_does_not_update_params(model, batch):
    before = [jnp.array(x) for x in jax.tree.leaves(nnx.state(model, nnx.Param))]
    validation_step(model, batch, jax.random.key(0))
    after = jax.tree.leaves(nnx.state(model, nnx.Param))
    assert all(jnp.allclose(b, a) for b, a in zip(before, after)), \
        "validation must not modify parameters"