import jax
import jax.numpy as jnp
from flax import struct

from bayesian_vae.models import BayesianVAE
from bayesian_vae.utils import PostLog


@struct.dataclass
class LossAux:
    reconstruction_loss: jax.Array
    latent_kl_divergence: jax.Array
    weight_kl_divergence: jax.Array


def erf_reconstruction_loss(mean, lnvar, target, bin_width: float=1/255):
    """Discretized-Gaussian NLL (nats) per image, summed over pixels.

    Computes log(Phi(upper) - Phi(lower)) in a numerically stable way:
    log_ndtr avoids CDF underflow in the tails, and the log1mexp-style
    difference avoids catastrophic cancellation when the two edges are
    close or both deep in a tail.
    """
    lnvar = jnp.clip(lnvar, -25.0, 25.0)
    inv_std = jnp.exp(-0.5 * lnvar)                 # 1/std, one exp, no divides
    half_bin = 0.5 * bin_width 
    # standardized bin edges (lower < upper)
    lower = (target - half_bin - mean) * inv_std
    upper = (target + half_bin - mean) * inv_std

    # Reflect into the left tail when the bin sits in the right tail,
    # so both log_ndtr evaluations run where they're most accurate.
    # Phi(u) - Phi(l) == Phi(-l) - Phi(-u), so swap+negate when l+u > 0.
    flip = (lower + upper) > 0.0
    a = jnp.where(flip, -upper, lower)              # smaller edge
    b = jnp.where(flip, -lower, upper)              # larger edge  (b >= a)

    log_cdf_a = jax.scipy.special.log_ndtr(a)                         # log Phi(a)
    log_cdf_b = jax.scipy.special.log_ndtr(b)                         # log Phi(b),  >= log_cdf_a

    # log(Phi(b) - Phi(a)) = log_cdf_b + log1mexp(log_cdf_a - log_cdf_b)
    delta = log_cdf_a - log_cdf_b                   # <= 0
    log_prob = log_cdf_b + _log1mexp(delta)

    nll = -log_prob                                 # nats, non-negative
    return jnp.mean(jnp.sum(nll, axis=(1, 2, 3)))   # [B, H, W, C]


def _log1mexp(x):
    """Stable log(1 - exp(x)) for x <= 0 (Mächler's two-branch form)."""
    # near 0: log(-expm1(x));  more negative: log1p(-exp(x))
    return jnp.where(
        x > -jnp.log(2.0),
        jnp.log(-jnp.expm1(x)),
        jnp.log1p(-jnp.exp(x)),
    )


def compute_training_loss(
    model: BayesianVAE,
    x: jax.Array,
    key: jax.Array,
    kl_weight_scale: jax.Array,
) -> tuple[jax.Array, LossAux, PostLog]:
    """Computes training loss of VAE.

    Args:
        x: Input image, shape [B, H, W, C_in]
        key: Input PRNG key, shape [1]
        kl_weight_scale: Rescales weights so that we have 1 weights Kl-div per epoch of training
            It is 1/n_images_per_batch, shape [1].
    """
    x_hat_mu, x_hat_lnvar, z_mu, z_lnvar, _ = model(x, key)
    batch_size = x.shape[0]
    reconstruction_loss = erf_reconstruction_loss(x_hat_mu, x_hat_lnvar, x)
    latent_kl_loss = (
        jnp.sum(model.calculate_latent_kl_divergence(z_mu, z_lnvar)) / batch_size
    )

    # Although weight KL-Div depends on weights, and weights are static for each element within
    # one mini-batch, we divide by batch size because we want to have weights kl loss affect our
    # stoch gradient descent ONCE PER EPOCH.
    weight_kl_loss = (
        kl_weight_scale
        * jnp.sum(model.calculate_vae_weights_kl_divergence())
        / batch_size
    )

    total_loss = reconstruction_loss + latent_kl_loss + weight_kl_loss

    return total_loss, LossAux(reconstruction_loss, latent_kl_loss, weight_kl_loss), PostLog(z_mu, z_lnvar)


def compute_validation_reconstruction_loss(
    model: BayesianVAE,
    x: jax.Array,
    key: jax.Array,
) -> jax.Array:
    x_hat_mu, x_hat_lnvar, *_ = model(x, key)
    return erf_reconstruction_loss(x_hat_mu, x_hat_lnvar, x)
