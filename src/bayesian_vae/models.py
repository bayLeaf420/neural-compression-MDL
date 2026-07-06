import jax
import jax.numpy as jnp
import flax.nnx as nnx

from bayesian_vae.layers import BayesianLinear, BayesianConv2D
from bayesian_vae.config import EncoderConfig, DecoderConfig, VaeConfig
from bayesian_vae.utils import PriorParam

MIN_VALID_DIM = 2

class BayesianEncoder(nnx.Module):
    """Bayesian Convolutional Encoder.

    Uses a convolution layers followed by linear layers to map input image batch to latents.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int],  # (height, width, in_channels)
        encoder_config: EncoderConfig,
        z_dim: int,
        prior_lnvar: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialises a Bayesian Encoder.

        Args:
            in_shape: Shape of input 
            encoder_config: Encoder hyperparams
            z_dim: Dimensions of latent variable
            prior_lnvar: See 'BayesianLinear'
            rngs: FLAX PRNG function.
        """
        super().__init__()
        self.z_dim = z_dim

        kernels = encoder_config.conv.kernels
        strides = encoder_config.conv.strides
        channels = encoder_config.conv.channels

        hidden_dims = encoder_config.lin.hidden_dims

        # ---- Convolution Stack ----
        self.conv_layers = nnx.List([]) 
        in_channels = in_shape[-1]
        for kernel_size, stride, out_channels in zip(kernels, strides, channels):
            self.conv_layers.append(
                BayesianConv2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding="SAME",
                    prior_lnvar=prior_lnvar,
                    rngs=rngs,
                )
            )
            in_channels = out_channels

        # Now we don't know output size.
        # We have to use jax.eval_shape
        dummy_x = jnp.zeros((1, *in_shape), dtype=jnp.float32)
        dummy_key_batch = jax.random.split(jax.random.key(0), 1)  # Shape (1,)

        def _run_conv_stack(x: jax.Array, key_batch: jax.Array) -> jax.Array:
            """Run the convolution stack.

            Args:
                x: Input image of shape [B, H, W, C_in]
                key_batch: Inputted PRNG keys of shape [B,]
            
            Returns:
                x: Output of shape [B, H, W, C_out_last]
            """
            for conv_layer in self.conv_layers:
                x = jax.nn.relu(conv_layer(x, key_batch))
            return x

        # jax.eval_shape outputs a struct, we need to convert it to a jax.Array before running jnp.prod
        conv_out = jax.eval_shape(_run_conv_stack, dummy_x, dummy_key_batch)
        conv_shape = jnp.asarray(conv_out.shape)

        h_out, w_out = conv_shape[1], conv_shape[2]
        if h_out < MIN_VALID_DIM or w_out < MIN_VALID_DIM:
            raise ValueError(
                f"Conv stack collapsed to {h_out}, {w_out}\n"
                f"too much downsampling for input {in_shape}."
            )

        flat_dim = int(jnp.prod(conv_shape[1:]))  # drop batch dim

        # ---- Dense stack ----
        self.lin_layers = nnx.List([])
        in_features = flat_dim

        for hidden_dim in hidden_dims:
            self.lin_layers.append(
                BayesianLinear(
                    in_features, hidden_dim, prior_lnvar=prior_lnvar, rngs=rngs
                )
            )
            in_features = hidden_dim

        # ---- Latent Distribution heads (also Bayesian) ----
        self.lin_last_mean = BayesianLinear(
            in_features, z_dim, prior_lnvar=prior_lnvar, rngs=rngs
        )
        self.lin_last_lnvar = BayesianLinear(
            in_features, z_dim, prior_lnvar=prior_lnvar, rngs=rngs
        )

    def __call__(
        self,
        x: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Perform Encoding of image.

        Args:
            x: Input image of shape [B, H, W, C_in]
            key: Inputted PRNG Key.

        Returns:
            z_mu: Mean of latent distribution of shape [B, z_dim]
            z_lnvar: Natural log-variance of latent distribution of shape [B, z_dim]
        """
        batch_size = x.shape[0]  # [B, H, W, C_in]

        # One key per Bayesian Layer. + 2 for final 2 linear layers
        num_layers = len(self.conv_layers) + len(self.lin_layers) + 2
        layer_keys = jax.random.split(key, num_layers)
        k_i = 0

        hidden = x
        for conv_layer in self.conv_layers:
            key_batch = jax.random.split(layer_keys[k_i], batch_size)
            hidden = jax.nn.relu(conv_layer(hidden, key_batch))
            k_i += 1

        hidden = hidden.reshape((batch_size, -1))  # Flatten (H, W, C) -> (H*W*C,)

        for lin_layer in self.lin_layers:
            key_batch = jax.random.split(layer_keys[k_i], batch_size)
            hidden = jax.nn.relu(lin_layer(hidden, key_batch))
            k_i += 1

        mean_keys = jax.random.split(layer_keys[k_i], batch_size)
        k_i += 1
        lnvar_keys = jax.random.split(layer_keys[k_i], batch_size)
        k_i += 1

        z_mu = self.lin_last_mean(hidden, mean_keys)
        z_lnvar = self.lin_last_lnvar(hidden, lnvar_keys)

        return z_mu, z_lnvar

    def calculate_kl_divergence(self) -> jax.Array:
        """Calculate KL divergence of weights.
        """
        total_kl = jnp.asarray(0.0)
        for conv_layer in self.conv_layers:
            total_kl += conv_layer.calculate_kl_divergence()
        for lin_layer in self.lin_layers:
            total_kl += lin_layer.calculate_kl_divergence()
        total_kl += self.lin_last_mean.calculate_kl_divergence()
        total_kl += self.lin_last_lnvar.calculate_kl_divergence()
        return total_kl


class BayesianDecoder(nnx.Module):
    """Latent Decoder

    Decodes a given latent into a probability distribution over an image, which can then be sampled.
    """

    def __init__(
        self,
        z_dim: int,
        decoder_config: DecoderConfig,
        output_shape: tuple[int, int, int],  # (H, W, C)
        prior_lnvar: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise Latent decoder.

        Args:
            z_dim: Dimensions of latent variable input.
            decoder_config: Decoder hyperparams.
            output_shape: Same as input image, [H, W, C_in]
            prior_lnvar: See 'BayesianLinear'
            rngs: See 'BayesianLinear'
        """
        super().__init__()

        self.output_shape = output_shape
        self.lin_layers = nnx.List([])

        hidden_dims = decoder_config.lin.hidden_dims
        in_features = z_dim
        for hidden_dim in hidden_dims:
            self.lin_layers.append(
                BayesianLinear(
                    in_features, hidden_dim, prior_lnvar=prior_lnvar, rngs=rngs
                )
            )
            in_features = hidden_dim

        flat_dim = int(output_shape[0] * output_shape[1] * output_shape[2])
        self.lin_last_mean = BayesianLinear(
            in_features, flat_dim, prior_lnvar=prior_lnvar, rngs=rngs
        )
        self.lin_last_lnvar = BayesianLinear(
            in_features, flat_dim, prior_lnvar=prior_lnvar, rngs=rngs
        )

    def __call__(self, z: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Decode a sampled latent into an image distribution.

        Args:
            z: Input latent of shape [B, z_dim]
            key: Inputted PRNG key.

            Returns:
            x_hat_mu: Mean of reconstructed image distribution, of shape [B, H, W, C_in]
            x_hat_lnvar: Natural log-variance of reconstructed image, [B, H, W, C_in]
        """
        batch_size = z.shape[0]  # shape is (B, z_dim)

        num_layers = len(self.lin_layers) + 2
        layer_keys = jax.random.split(key, num_layers)
        k_i = 0

        hidden = z
        for lin_layer in self.lin_layers:
            key_batch = jax.random.split(layer_keys[k_i], batch_size)
            hidden = jax.nn.relu(lin_layer(hidden, key_batch))
            k_i += 1

        key_batch = jax.random.split(layer_keys[k_i], batch_size)
        x_hat_mu = self.lin_last_mean(hidden, key_batch)
        k_i += 1

        key_batch = jax.random.split(layer_keys[k_i], batch_size)
        x_hat_lnvar = self.lin_last_lnvar(hidden, key_batch)

        x_hat_mu = x_hat_mu.reshape((batch_size, *self.output_shape))
        x_hat_lnvar = x_hat_lnvar.reshape((batch_size, *self.output_shape))

        return x_hat_mu, x_hat_lnvar

    def calculate_kl_divergence(self) -> jax.Array:
        """See 'BayesianEncoder'."""
        total_kl = jnp.asarray(0.0)
        for lin_layer in self.lin_layers:
            total_kl += lin_layer.calculate_kl_divergence()
        total_kl += self.lin_last_mean.calculate_kl_divergence()
        total_kl += self.lin_last_lnvar.calculate_kl_divergence()
        return total_kl


class BayesianVAE(nnx.Module):
    """Full VAE for creating an image generator."""
    def __init__(
        self,
        in_shape: tuple[int, int, int],  # (height, width, channels)
        config: VaeConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise encoder and decoder.
        
        Args:
            in_shape: [H, W, C_in]
            config: Contains 'encoder_config', 'decoder_config', 'z_prior_lnvar', 'w_prior_lnvar',
                'z_dim'.
            rngs: See 'BayesianEncoder'
        """
        super().__init__()

        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        z_dim = config.z_dim
        self.z_free_nats = config.z_free_nats
        w_prior_lnvar = config.w_prior_lnvar

        self.encoder = BayesianEncoder(
            in_shape,
            encoder_config,
            z_dim,
            w_prior_lnvar,
            rngs=rngs,
        )

        output_shape = in_shape
        self.decoder = BayesianDecoder(
            z_dim,
            decoder_config,
            output_shape,
            w_prior_lnvar,
            rngs=rngs,
        )

        # Prior will be a gaussian. Initialised as N(0, I).
        self.z_prior_mu = PriorParam(jnp.zeros((z_dim,)))
        self.z_prior_lnvar = PriorParam(jnp.zeros((z_dim,)))

    def __call__(
        self,
        x: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, ...]:
        """Run full Bayesian VAE

        Args:
            x: Input image of shape [B, H, W, C_in]
            key: Inputted PRNG key.

        Returns:
            x_hat_mu: Mean of reconstructed image dist, [B, H, W, C_in]
            x_hat_lnvar: Natural log-variance of reconstructed image dist, [B, H, W, C_in]
            z_mu: Mean of latent distribution, [B, z_dim]
            z_lnvar: Natural log-variance of latent distribution
            z: Sampled latent [B, z_dim]
        """

        enc_key, sampling_key, dec_key = jax.random.split(key, 3)
        z_mu, z_lnvar = self.encoder(x, enc_key)  # Each [B, z_dim]

        
        def _single_example_latent_sample(example_key, z_mu, z_lnvar):
            """To be Vmapped function decoding a single latent.

            Args:
              key (jax.Array): [1]
              z_mu (jax.Array): [z_dim]
              z_lnvar (jax.Array): [z_dim]
            """
            noise = jax.random.normal(example_key, z_mu.shape)
            return z_mu + noise * jnp.exp(0.5 * z_lnvar)

        z_key_batch = jax.random.split(sampling_key, x.shape[0])
        z = jax.vmap(_single_example_latent_sample, in_axes=(0, 0, 0))(
            z_key_batch, z_mu, z_lnvar
        )
        x_hat_mu, x_hat_lnvar = self.decoder(z, dec_key)
        return x_hat_mu, x_hat_lnvar, z_mu, z_lnvar, z

    def calculate_latent_kl_divergence(self, z_mu, z_lnvar):
        """Calculate KL divergence of latent w.r.t. the same prior_lnvar as weights for simplicity.
        
        Args:
            z_mu: Mean of latent posterior distribution
            z_lnvar: ln(variance) of latent posterior distribution.
        """
        def _elementwise_latent_kl(z_mu, z_lnvar):
            elementwise_kl = 0.5 * (self.z_prior_lnvar - z_lnvar 
                            + jnp.exp(z_lnvar - self.z_prior_lnvar)
                            + jnp.exp(-self.z_prior_lnvar) * (z_mu - self.z_prior_mu)**2) - 0.5
            floored_kl = jnp.maximum(jnp.asarray(self.z_free_nats), elementwise_kl)
            return floored_kl

        def _single_example_latent_kl_divergence(z_mu, z_lnvar):
            """For single sample, to be jax.vmap-ped over batch"""
            return jnp.sum(_elementwise_latent_kl(z_mu, z_lnvar))

        return jax.vmap(_single_example_latent_kl_divergence, in_axes=(0, 0))(
            z_mu, z_lnvar,
        )

    def calculate_vae_weights_kl_divergence(self):
        """Calculate final KL divergence to be used during training."""
        return (
            self.encoder.calculate_kl_divergence()
            + self.decoder.calculate_kl_divergence()
        )
