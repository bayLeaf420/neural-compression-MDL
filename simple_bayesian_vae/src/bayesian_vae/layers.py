import jax
import jax.numpy as jnp
import flax.nnx as nnx


def gaussian_kl_divergence(
    post_mean: jax.Array,
    post_lnvar: jax.Array,
    prior_lnvar: float,
) -> jax.Array:
    """
    Helper function to calculate KL-divergence of gaussian posterior with respect to prior. The
    posterior is the 'approximating' distribution and the prior is the assumed 'true' distribution.

    Args:
      post_mean: Input array of shape (L,) for some L.
      post_lnvar: Input array of shape (L,), representing the natural log of diagonal of posterior
        covariance matrix.
      prior_lnvar: Input float, represents prior log, jax.numpy handles broadcasting it for calculation.

    Returns:
      jax.Array: Scalar representing D_kl(post || prior).
    """
    post_var = jnp.exp(post_lnvar)
    prior_var = jnp.exp(prior_lnvar)
    elementwise_kl = 0.5 * (
        jnp.log(prior_var / post_var) + (post_var + post_mean**2) / prior_var - 1.0
    )
    return jnp.sum(elementwise_kl)


class BayesianLinear(nnx.Module):
    """A Bayesian linear layer with weight and bias uncertainty.

    Uses reparameterization trick to sample weights and biases from learned distributions
    during forward pass.
    """

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        use_bias: bool = True,
        prior_lnvar: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise a Bayesian linear layer.

        Args:
            in_dims: Number of dims possessed by input
            out_dims: Number of dims possessed by output
            use_bias: Whether to include bias.
            prior_lnvar: Log-variance of prior distribution.
            rngs: FLAX/JAX random number generator.
        """

        super().__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.prior_lnvar = prior_lnvar
        self.use_bias = use_bias

        # rngs.params() returns a fresh deterministic key each time its called.
        # We initialise mean with a normal distribution(0, 0.05)
        self.w_mu = nnx.Param(
            jax.random.normal(rngs.params(), (in_dims, out_dims)) * 0.05
        )

        # We initialise weight lnvar the same way
        self.w_lnvar = nnx.Param(jnp.full((in_dims, out_dims), -5.0))

        if self.use_bias:
            # We initialise biases with mean 0.0 and lnvar 0.0
            self.b_mu = nnx.Param(jnp.zeros((out_dims,)))
            self.b_lnvar = nnx.Param(jnp.full((out_dims,), -5.0))
        else:
            self.b_mu = None
            self.b_lnvar = None

    def __call__(
        self, x: jax.Array, key_batch: jax.Array
    ) -> jax.Array:  # x: [B, din], key_batch: [B, din]
        """
        Args:
          inputs (jax.Array): [batch_size, in_dims]
          key_batch (jax.Array): [batch_size, 2], one independent key per example
            produced by jax.random.split(key, batch_size)

        Returns:
          jax.Array: [batch_size, out_dims]
        """
        w_mu = self.w_mu
        w_lnvar = self.w_lnvar
        b_mu = self.b_mu  # is None if self.use_bias is False
        b_lnvar = self.b_lnvar

        def _single_example_forward(
            key: jax.Array,
            x: jax.Array,
        ) -> jax.Array:
            """Single argument forward pass. Vmap this to operate on a batch.

            Args:
              key (jax.Array): [1]
              x (jax.Array): [in_dims]

            Returns:
              jax.Array: [out_dims]
            """
            w_key, b_key = jax.random.split(key)

            w_noise = jax.random.normal(w_key, w_mu.shape)
            w = w_mu + w_noise * jnp.exp(0.5 * w_lnvar)

            out = x @ w
            if self.use_bias:
                b_noise = jax.random.normal(b_key, b_mu.shape) # type: ignore
                b = b_mu + b_noise * jnp.exp(0.5 * b_lnvar) # type: ignore
                out += b

            return out

        return jax.vmap(_single_example_forward, in_axes=(0, 0))(key_batch, x)

    def calculate_kl_divergence(self) -> jax.Array:
        """Calculate KL divergence of weight distribution w.r.t. prior distribution."""
        total_kl = gaussian_kl_divergence(
            self.w_mu.value,
            self.w_lnvar.value,
            self.prior_lnvar,
        )
        if self.use_bias:
            total_kl += gaussian_kl_divergence(
                self.b_mu.value, # type: ignore -> FORMATTING WHOLE REPO/THIS FILE REMOVES THIS COMMENT, 
                self.b_lnvar.value, # type: ignore
                self.prior_lnvar,
            )
        return total_kl


class BayesianConv2D(nnx.Module):
    """Bayesian Convolution layer. 

    Uses reparametrisation trick to find posterior distribution over convolution kernels and biases.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        strides: tuple[int, int] = (1, 1),
        padding: str | tuple[tuple[int, int], tuple[int, int]] = "SAME",
        input_dilation: tuple[int, int] = (1, 1),
        kernel_dilation: tuple[int, int] = (1, 1),
        feature_group_count: int = 1,
        use_bias: bool = True,
        prior_lnvar: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise Bayesian convolution layer. 

        Args:
            in_channels: Number of channels in input.
            out_channels: Number of output channels.
            kernel_size: Size of kernel which broadcasts over the input matrix.
            strides: Tuple containing vertical and horizontal stride, i.e. how much the kernel jumps by. 
            padding: How the kernel handles edges, "SAME"==>Output images will have same height and width. 
            input_dilation: Not of use for us, just there to satisfy jax.lax.conv_general_dilated
            kernel_dilation: There to satisfy jax.lax.conv_general_dilated
            feature_group_count: Groups channels together (?)
            use_bias: obvious
            prior_lnvar: Natural log-variance of prior distribution over weights. 

        Returns: 
            Nothing, just initializes a BayesianConv2D object.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.input_dilation = input_dilation
        self.kernel_dilation = kernel_dilation
        self.feature_group_count = feature_group_count
        self.use_bias = use_bias
        self.prior_lnvar = prior_lnvar

        if in_channels % feature_group_count != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by feature_group_count ({feature_group_count})"
            )
        if out_channels % feature_group_count != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by feature_group_count ({feature_group_count})"
            )

        # kernel shape: (height, width, in_channels, out_channels)
        kernel_shape = (
            *kernel_size,
            in_channels // feature_group_count,
            out_channels,
        )
        bias_shape = (out_channels,)

        self.kernel_mean = nnx.Param(
            jax.random.normal(rngs.params(), kernel_shape) * 0.05
        )
        self.kernel_lnvar = nnx.Param(jnp.full(kernel_shape, -5.0))

        if use_bias:
            self.b_mu = nnx.Param(jnp.zeros(bias_shape))
            self.b_lnvar = nnx.Param(jnp.full(bias_shape, -5.0))
        else:
            self.b_mu = None 
            self.b_lnvar = None

    def __call__(
        self,
        x: jax.Array,
        key_batch: jax.Array,
    ) -> jax.Array:
        """
        Args:
          x (jax.Array): [batch_size, height, width, in_channels], input image batch.
          key_batch (jax.Array): [batch_size, 2], one independent key per example
            produced by jax.random.split(key, batch_size), batch of PRNG Keys.

        Returns:
          jax.Array: [batch_size, height_out, width_out, out_channels], output of convolution.
        """
        kernel_mean = self.kernel_mean
        kernel_lnvar = self.kernel_lnvar
        b_mu = self.b_mu  # If use_bias = False, is None
        b_lnvar = self.b_lnvar

        def _single_example_forward(key, x):
            """
            Args:
              key (jax.Array): [1]
              x (jax.Array): [height, width, in_channels]

            Returns:
              jax.Array: [height_out, width_out, out_channels]
            """
            kernel_noise_key, b_key = jax.random.split(key)
            w_noise = jax.random.normal(kernel_noise_key, kernel_mean.shape)

            w = kernel_mean + w_noise * jnp.exp(0.5 * kernel_lnvar)

            # Expand to include batch dimension as that is needed by conv_general_dilated
            x = x[jnp.newaxis, ...]  # [1, H, W, C_in]
            out = jax.lax.conv_general_dilated(
                lhs=x,
                rhs=w,
                window_strides=self.strides,
                padding=self.padding,
                lhs_dilation=self.input_dilation,
                rhs_dilation=self.kernel_dilation,
                feature_group_count=self.feature_group_count,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )  # shape: (1, H_out, W_out, out_channels)

            if self.use_bias:
                b_noise = jax.random.normal(b_key, b_mu.shape) # type: ignore
                b = b_mu + b_noise * jnp.exp(0.5 * b_lnvar) # type: ignore
                out = out + b.reshape(1, 1, 1, -1)
                # Bias has shape (out_channels,) -> Broadcasts to shape (1, H_out, W_out, C_out)

            return out[0]  # Remove batch dimension

        return jax.vmap(_single_example_forward, in_axes=(0, 0))(key_batch, x)

    def calculate_kl_divergence(self) -> jax.Array:
        """Calculate KL divergence of weight distribution w.r.t. prior distribution."""
        total_kl = gaussian_kl_divergence(
            self.kernel_mean.value,
            self.kernel_lnvar.value,
            self.prior_lnvar,
        )
        if self.use_bias:
            total_kl += gaussian_kl_divergence(
                self.b_mu.value, # type: ignore
                self.b_lnvar.value, # type: ignore
                self.prior_lnvar,
            )
        return total_kl
