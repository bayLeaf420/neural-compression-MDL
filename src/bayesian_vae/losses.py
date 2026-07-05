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


def gaussian_reconstruction_loss(
    mean: jax.Array,
    lnvar: jax.Array,
    target: jax.Array,
) -> jax.Array:
    """
    Computes the per-pixel negative log-likelihood for a Gaussian distribution.

    Args:
      mean: [batch_size, height, width, channels]
      lnvar: [batch_size, height, width, channels]
      target: [batch_size, height, width, channels]

    Returns:
        Average NLL over the batch and all pixels.
    """
    nll_per_pixel = 0.5 * lnvar + 0.5 * jnp.exp(-lnvar) * (target - mean) ** 2
    # Sum over spatial dimensions then avg over batch
    return jnp.mean(jnp.sum(nll_per_pixel, axis=(1, 2, 3)))


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
    reconstruction_loss = gaussian_reconstruction_loss(x_hat_mu, x_hat_lnvar, x)
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
    return gaussian_reconstruction_loss(x_hat_mu, x_hat_lnvar, x)
