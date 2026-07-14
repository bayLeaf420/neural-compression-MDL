import jax
import jax.numpy as jnp
import flax.nnx as nnx


from bayesian_vae.newlayers import NewLin as BayesianLinear
from bayesian_vae.newlayers import NewConv as BayesianConv2D


class BayesVAE(nnx.Module):
    """Simple Bayesian VAE"""


    def __init__(self, z_dim=48, mode: str='test', z_prior_lnvar=0.0, z_free_nats = 0.5, *, rngs):
        """Initialize the 319,776 * 2 params of encoder"""

        self.z_prior_lnvar = jnp.asarray(z_prior_lnvar) 
        self.z_free_nats = jnp.asarray(z_free_nats)


        ### --- Encoder --- ###
        # [B, 28, 28, 1] -> [B, 14, 14, 32], prod=6272
        self.enc_conv1 = BayesianConv2D(1, 32, (3, 3), strides=(2, 2), rngs=rngs)
        # [B, 14, 14, 32] -> [B, 7, 7, 64], prod=3136
        self.enc_conv2 = BayesianConv2D(32, 64, (3, 3), (2, 2), rngs=rngs) 
        # [B, 3136] -> [B, 48]
        self.enc_lin_mu = BayesianLinear(64 * 7 * 7, z_dim, rngs=rngs)
        self.enc_lin_lnvar = BayesianLinear(64 * 7 * 7, z_dim, rngs=rngs)

        ### --- Latent --- ###
        self.z_dim = z_dim 

        ### --- Decoder --- ###

        # After sampling z ~ N(z_mu, z_var)
        self.dec_lin1 = BayesianLinear(z_dim, 64 * 7 * 7, rngs=rngs)
        # After reshaping and [B, 7, 7, 32] -> [B, 14, 14, 32] resize do:
        self.dec_conv1 = BayesianConv2D(64, 32, (3, 3), rngs=rngs) # stride is 1 by def
        # After [B, 14, 14, 32] -> [B, 28, 28, 32] resize do:
        self.dec_conv2_mu = BayesianConv2D(32, 1,  (3, 3), rngs=rngs)
        self.dec_conv2_lnvar = BayesianConv2D(32, 1, (3, 3), rngs=rngs)


    def encode(self, x, key):
        """Encode image batch into laent distribution"""
        keys = jax.random.split(key, (4, x.shape[0]))
        h = nnx.relu(self.enc_conv1(x, keys[0]))
        h = nnx.relu(self.enc_conv2(h, keys[1]))
        h = h.reshape(h.shape[0], -1)
        return self.enc_lin_mu(h, keys[2]), self.enc_lin_lnvar(h, keys[3])


    def sample_latent(self, z_mu, z_lnvar, key):
        """Sample a latent from posterior"""
        z_eps = jax.random.normal(key, (z_mu.shape[0], self.z_dim))
        z = z_mu + z_eps * jnp.exp(0.5 * z_lnvar)
        return z


    def decode(self, z, key):
        keys = jax.random.split(key, (4, z.shape[0]))
        h = self.dec_lin1(z, keys[0]).reshape(-1, 7, 7, 64)
        h = jax.image.resize(h, (h.shape[0], 14, 14, 64), 'nearest')
        h = nnx.relu(self.dec_conv1(h, keys[1]))
        h = jax.image.resize(h, (h.shape[0], 28, 28, 32), 'nearest')
        return self.dec_conv2_mu(h, keys[2]), self.dec_conv2_lnvar(h, keys[3])
        

    def __call__(self, x, key, mode: str='train'):
        enc_key, samp_key, dec_key = jax.random.split(key, 3)
        z_mu, z_lnvar = self.encode(x, enc_key)

        assert mode in ('train', 'test', 'inference'), f"Expected mode = 'train', 'test', or 'inference', but got {mode}"

        if mode in ('train', 'inference'):
            z = self.sample_latent(z_mu, z_lnvar, samp_key)
        elif mode == 'test':
            z = z_mu 
        
        x_hat_mu, x_hat_lnvar = self.decode(z, dec_key)

        return x_hat_mu, x_hat_lnvar, z_mu, z_lnvar
    

    def calc_latent_kl(self, z_mu, z_lnvar):
        """Calculate average latent KL over batch"""
        # Both of shape [B, z_dim]
        kl = 0.5 * (self.z_prior_lnvar - z_lnvar
                    + jnp.exp(z_lnvar - self.z_prior_lnvar)
                    + jnp.exp(-self.z_prior_lnvar) * z_mu ** 2) - 0.5
        kl = jnp.maximum(kl, self.z_free_nats)
        # Just sum over everything - clean, handle batch size and stuff in train script
        return jnp.sum(kl, axis=1) 
    

    def calc_param_kl(self):
        kl = jnp.asarray(0.0)
        kl += self.enc_conv1.calculate_kl_divergence()
        kl += self.enc_conv2.calculate_kl_divergence()
        kl += self.enc_lin_mu.calculate_kl_divergence()
        kl += self.enc_lin_lnvar.calculate_kl_divergence()
        kl += self.dec_lin1.calculate_kl_divergence()
        kl += self.dec_conv1.calculate_kl_divergence()
        kl += self.dec_conv2_mu.calculate_kl_divergence()
        kl += self.dec_conv2_lnvar.calculate_kl_divergence()
        return kl 



        

        

        

