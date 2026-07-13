import jax
import jax.numpy as jnp
from flax import struct

from bayesian_vae.newmodel import BayesVAE
from bayesian_vae.overmodel import OverVAE
from bayesian_vae.newlosses import erf_reconstruction_loss


@struct.dataclass
class OverLossAux:
    reconstruction_loss: jax.Array   # summed over batch (nats)
    latent_kl: jax.Array             # summed over batch (nats)
    weight_nll: jax.Array            # ONCE for the batch (nats)


def base_batch_loss(model: BayesVAE, x: jax.Array, key: jax.Array) -> jax.Array:
    """Base model description length for a batch (nats): SUM over images of
    (recon + latent_kl). NO weight term (base weights are the shared prior,
    not transmitted per-batch). Deterministic reconstruction."""
    x_hat_mu, x_hat_lnvar, z_mu, z_lnvar = model(x, key, mode='test')
    # erf_reconstruction_loss returns MEAN over batch -> multiply back to SUM
    B = x.shape[0]
    recon = erf_reconstruction_loss(x_hat_mu, x_hat_lnvar, x) * B
    latent_kl = model.calc_latent_kl(z_mu, z_lnvar) * B   # calc_latent_kl is per-image mean -> sum
    return recon + latent_kl


def over_mdl_loss(model: OverVAE, x: jax.Array) -> tuple[jax.Array, OverLossAux]:
    """ABSOLUTE MDL description length for a batch (nats), no lambda scaling:

        sum_i[recon(x_i) + latent_kl(x_i)]  +  weight_NLL   (weight cost ONCE)

    This is the true number of nats to transmit the 80 images under the
    overfit-and-quantized weights plus the weights themselves.
    """
    x_hat_mu, x_hat_lnvar, z_mu, z_lnvar = model(x)        # deterministic, no key
    B = x.shape[0]
    recon = erf_reconstruction_loss(x_hat_mu, x_hat_lnvar, x) * B   # MEAN -> SUM
    latent_kl = model.calc_latent_kl(z_mu, z_lnvar) * B            # per-image mean -> SUM
    weight_nll = model.calculate_sampling_nll()                    # ONCE, absolute, no scale
    total = recon + latent_kl + weight_nll
    return total, OverLossAux(recon, latent_kl, weight_nll)