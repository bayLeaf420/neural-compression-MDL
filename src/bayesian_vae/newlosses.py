import jax
import jax.numpy as jnp
from flax import struct

from bayesian_vae.newmodel import BayesVAE
from bayesian_vae.utils import PostLog


@struct.dataclass
class LossAux:
    reconstruction_loss: jax.Array
    latent_kl_divergence: jax.Array
    weight_kl_divergence: jax.Array


def erf_reconstruction_loss(
    mean, lnvar, target,
    bin_width: float = 1/255,
    low_edge: float = 0.0,
    high_edge: float = 1.0,
):
    """Discretized-Gaussian NLL (nats) per image, summed over pixels.

    Interior bins integrate the Gaussian over [target - half_bin, target + half_bin].
    The lowest bin (target == low_edge) integrates (-inf, low_edge + half_bin],
    and the highest bin (target == high_edge) integrates [high_edge - half_bin, +inf),
    so saturated pixels get their full tail mass (PixelCNN++-style edge handling).
    """
    lnvar = jnp.clip(lnvar, -25.0, 25.0)
    inv_std = jnp.exp(-0.5 * lnvar)
    half_bin = 0.5 * bin_width

    lower = (target - half_bin - mean) * inv_std
    upper = (target + half_bin - mean) * inv_std

    # --- interior: stable log(Phi(upper) - Phi(lower)) ---
    flip = (lower + upper) > 0.0
    a = jnp.where(flip, -upper, lower)              # smaller edge
    b = jnp.where(flip, -lower, upper)              # larger edge (b >= a)
    log_cdf_a = jax.scipy.special.log_ndtr(a)
    log_cdf_b = jax.scipy.special.log_ndtr(b)
    delta = log_cdf_a - log_cdf_b                   # <= 0
    log_prob_interior = log_cdf_b + _log1mexp(delta)

    # --- boundary bins ---
    # lowest level: mass from -inf to upper  ->  log Phi(upper) = log_ndtr(upper)
    log_prob_low = jax.scipy.special.log_ndtr(upper)
    # highest level: mass from lower to +inf -> log(1 - Phi(lower)) = log_ndtr(-lower)
    log_prob_high = jax.scipy.special.log_ndtr(-lower)

    # --- select per pixel ---
    at_low = target <= low_edge + half_bin          # pixels sitting in the lowest bin
    at_high = target >= high_edge - half_bin         # pixels sitting in the highest bin
    log_prob = jnp.where(
        at_low, log_prob_low,
        jnp.where(at_high, log_prob_high, log_prob_interior),
    )

    nll = -log_prob                                  # nats, non-negative
    return jnp.mean(jnp.sum(nll, axis=(1, 2, 3)))    # [B, H, W, C]


def _log1mexp(x):
    """Stable log(1 - exp(x)) for x <= 0 (Mächler's two-branch form)."""
    # near 0: log(-expm1(x));  more negative: log1p(-exp(x))
    return jnp.where(
        x > -jnp.log(2.0),
        jnp.log(-jnp.expm1(x)),
        jnp.log1p(-jnp.exp(x)),
    )


def compute_training_loss(
    model: BayesVAE,
    x: jax.Array,
    key: jax.Array,
    kl_weight_scale: jax.Array,
    mode: str,
) -> tuple[jax.Array, LossAux]:
    """Computes training loss of VAE.

    Args:
        x: Input image, shape [B, H, W, C_in]
        key: Input PRNG key, shape [1]
        kl_weight_scale: Rescales weights so that we have 1 weights Kl-div per epoch of training
            It is 1/n_images_per_batch, shape [1].
    """
    x_hat_mu, x_hat_lnvar, z_mu, z_lnvar = model(x, key, mode=mode)
    batch_size = x.shape[0]
    reconstruction_loss = erf_reconstruction_loss(x_hat_mu, x_hat_lnvar, x)
    latent_kl_loss = (
        jnp.sum(model.calc_latent_kl(z_mu, z_lnvar)) / batch_size
    )

    # Although weight KL-Div depends on weights, and weights are static for each element within
    # one mini-batch, we divide by batch size because we want to have weights kl loss affect our
    # stoch gradient descent ONCE PER EPOCH.
    weight_kl_loss = (
        kl_weight_scale
        * jnp.sum(model.calc_param_kl())
        / batch_size
    )

    total_loss = reconstruction_loss + latent_kl_loss + weight_kl_loss

    return total_loss, LossAux(reconstruction_loss, latent_kl_loss, weight_kl_loss)


def compute_validation_reconstruction_loss(
    model: BayesVAE,
    x: jax.Array,
    key: jax.Array,
    mode: str,
) -> jax.Array:
    x_hat_mu, x_hat_lnvar, *_ = model(x, key, mode=mode)
    return erf_reconstruction_loss(x_hat_mu, x_hat_lnvar, x)
