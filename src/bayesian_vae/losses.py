import jax
import jax.numpy as jnp
from flax import struct

from .models import BayesianVAE
from .utils import PostLog


@struct.dataclass
class LossAux:
    reconstruction_loss: jax.Array
    latent_kl_divergence: jax.Array
    weight_kl_divergence: jax.Array


def erf_reconstruction_loss(mean, lnvar, target, num_bins=256):
    lnvar = jnp.clip(lnvar, -12.0, 12.0)
    std = jnp.exp(0.5 * lnvar)
    half_bin = 0.5 / (num_bins - 1)      # half the bin width, on [0,1] scale
    # standardized upper/lower bin edges
    upper = (target + half_bin - mean) / std
    lower = (target - half_bin - mean) / std
    # normal CDF via erf
    def cdf(z):
        return 0.5 * (1.0 + jax.lax.erf(z / jnp.sqrt(2.0)))
    prob = cdf(upper) - cdf(lower)                 # probability mass in the bin
    prob = jnp.clip(prob, 1e-12, 1.0)              # avoid log(0)
    nll = -jnp.log(prob)                           # non-negative, in NATS
    return jnp.mean(jnp.sum(nll, axis=(1, 2, 3))) # [B, H, W, C]


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
