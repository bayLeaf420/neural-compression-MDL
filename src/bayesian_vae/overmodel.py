import jax
import jax.numpy as jnp
from flax import nnx

from bayesian_vae.newmodel import BayesVAE
from bayesian_vae.overlayers import OverLin, OverConv, OverParam


class OverVAE(nnx.Module):
    """Overfitting/QAT VAE built FROM a trained BayesVAE.

    Same architecture as BayesVAE, but every Bayesian sublayer is replaced by
    an Over-layer whose weight is (a) initialised from the trained base mean,
    (b) fake-quantized in the forward pass, and (c) the ONLY trainable state
    (OverParam). The base (mu, lnvar) are frozen inside each Over-layer and
    used as the coding prior for the sampling-NLL / MDL term.

    Forward pass is deterministic (z = z_mu, no latent sampling, no keys):
    this network overfits a single fixed code, per the C3/Cool-chic style
    per-instance compression setup.
    """

    def __init__(self, base: BayesVAE, bits: int = 8):
        # carry over latent config / hyperparams from the base
        self.z_dim = base.z_dim
        self.z_prior_lnvar = base.z_prior_lnvar
        self.z_free_nats = base.z_free_nats

        # --- Encoder ---
        self.over_enc_conv1 = OverConv(base.enc_conv1, bits=bits)
        self.over_enc_conv2 = OverConv(base.enc_conv2, bits=bits)
        self.over_enc_lin_mu = OverLin(base.enc_lin_mu, bits=bits)
        self.over_enc_lin_lnvar = OverLin(base.enc_lin_lnvar, bits=bits)

        # --- Decoder ---
        self.over_dec_lin1 = OverLin(base.dec_lin1, bits=bits)
        self.over_dec_conv1 = OverConv(base.dec_conv1, bits=bits)
        self.over_dec_conv2_mu = OverConv(base.dec_conv2_mu, bits=bits)
        self.over_dec_conv2_lnvar = OverConv(base.dec_conv2_lnvar, bits=bits)

        # convenient list for the aggregate walks
        self._over_layers = (
            self.over_enc_conv1,
            self.over_enc_conv2,
            self.over_enc_lin_mu,
            self.over_enc_lin_lnvar,
            self.over_dec_lin1,
            self.over_dec_conv1,
            self.over_dec_conv2_mu,
            self.over_dec_conv2_lnvar,
        )

    def encode(self, x):
        # No keys: Over-layers are deterministic (quantized mean weights).
        h = nnx.relu(self.over_enc_conv1(x))
        h = nnx.relu(self.over_enc_conv2(h))
        h = h.reshape(h.shape[0], -1)
        return self.over_enc_lin_mu(h), self.over_enc_lin_lnvar(h)

    def decode(self, z):
        h = self.over_dec_lin1(z).reshape(-1, 7, 7, 64)
        h = jax.image.resize(h, (h.shape[0], 14, 14, 64), "nearest")
        h = nnx.relu(self.over_dec_conv1(h))
        h = jax.image.resize(h, (h.shape[0], 28, 28, 32), "nearest")
        return self.over_dec_conv2_mu(h), self.over_dec_conv2_lnvar(h)

    def __call__(self, x):
        z_mu, z_lnvar = self.encode(x)
        z = z_mu                                   # deterministic: no sampling
        x_hat_mu, x_hat_lnvar = self.decode(z)
        return x_hat_mu, x_hat_lnvar, z_mu, z_lnvar

    # ---- QAT plumbing ----

    def calibrate_all(self):
        """Refresh every sublayer's quant grid from current weights.
        Call OUTSIDE the gradient step, before each forward."""
        for layer in self._over_layers:
            layer.calibrate()

    def calculate_sampling_nll(self):
        """Total bits to code all quantized overfit weights under the frozen
        base distributions — the MDL / weight-rate term of the objective."""
        total = jnp.asarray(0.0)
        for layer in self._over_layers:
            total += layer.calculate_sampling_nll()
        return total

    def calc_latent_kl(self, z_mu, z_lnvar):
        """Same free-nats latent KL as BayesVAE (deterministic z, but the
        posterior params still get a KL-to-prior term)."""
        kl = 0.5 * (self.z_prior_lnvar - z_lnvar
                    + jnp.exp(z_lnvar - self.z_prior_lnvar)
                    + jnp.exp(-self.z_prior_lnvar) * z_mu ** 2) - 0.5
        kl = jnp.maximum(kl, self.z_free_nats)
        return jnp.mean(jnp.sum(kl, axis=1), axis=0)