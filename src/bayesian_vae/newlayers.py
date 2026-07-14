from flax import nnx 
import jax.numpy as jnp
import jax

from bayesian_vae.layers import BayesianLinear, BayesianConv2D
from bayesian_vae.overlayers import fake_quant, _calibrate_tensor, QScale, QZeroPoint

class NewLin(BayesianLinear):
    def __init__(self, in_dims, out_dims, use_bias=True, bits: int=8, *, rngs: nnx.Rngs):
        super().__init__(in_dims, out_dims, use_bias=use_bias, rngs=rngs)
        self.qmin = -(2 ** (bits - 1))
        self.qmax = (2 ** (bits - 1)) - 1
        self.w_scale = QScale(jnp.asarray(1.0))
        self.w_zero = QZeroPoint(jnp.asarray(0.0))
        if self.use_bias:
            self.b_scale = QScale(jnp.asarray(1.0))
            self.b_zero = QZeroPoint(jnp.asarray(0.0))

    def calibrate(self):
        s, z = _calibrate_tensor(self.w_mu[...], self.qmin, self.qmax)
        self.w_scale[...], self.w_zero[...] = s, z 
        if self.use_bias:
            s, z = _calibrate_tensor(self.b_mu[...], self.qmin, self.qmax) # type: ignore 
            self.b_scale[...], self.b_zero[...] = s, z 


    def __call__(self, x, key):
        # breakpoint()
        w_mu = fake_quant(self.w_mu[...], self.w_scale ,self.w_zero, self.qmin, self.qmax)      
        out_mu = x @ w_mu 
        w_lnvar = self.w_lnvar[...]
        out_var = x ** 2 @ jnp.exp(w_lnvar) 

        w_key, b_key = jax.random.split(key)
        out_noise = jax.random.normal(w_key, (x.shape[0], self.out_dims))
        
        out = out_mu + out_noise * jnp.sqrt(out_var + 1e-8)
        
        if self.use_bias:
            b_mu = fake_quant(self.b_mu[...], self.b_scale, self.b_zero, self.qmin, self.qmax) # type: ignore
            b_lnvar = self.b_lnvar[...] # type: ignore
            b_noise = jax.random.normal(b_key, (x.shape[0], self.out_dims))
            b = b_mu + b_noise * jnp.exp(0.5 * b_lnvar)
            out += b 
        
        return out 
    

class NewConv(BayesianConv2D):
    def __init__(
        self, in_channels, out_channels, kernel_size, 
        strides=(1, 1), feature_group_count=1, use_bias=True, bits: int=8, *, rngs: nnx.Rngs,
    ):
        super().__init__(in_channels, out_channels, kernel_size, strides=strides, feature_group_count=feature_group_count,  rngs=rngs)
        self.qmin = -(2 ** (bits - 1))
        self.qmax = (2 ** (bits - 1)) - 1
        self.kernel_scale = QScale(jnp.asarray(1.0))
        self.kernel_zero = QZeroPoint(jnp.asarray(0.0))
        if self.use_bias:
            self.b_scale = QScale(jnp.asarray(1.0))
            self.b_zero = QZeroPoint(jnp.asarray(0.0))
    
    
    def calibrate(self):
        s, z = _calibrate_tensor(self.kernel_mu[...], self.qmin, self.qmax)
        self.kernel_scale[...], self.kernel_zero[...] = s, z
        if self.use_bias:
            s, z = _calibrate_tensor(self.b_mu[...], self.qmin, self.qmax) # type: ignore
            self.b_scale[...], self.b_zero[...] = s, z

    def __call__(self, x, key):
        breakpoint()
        key_batch = jax.random.split(key, (x.shape[0],))
        kernel_mu = self.kernel_mu[...]
        kernel_mu = fake_quant(kernel_mu, self.kernel_scale, self.kernel_zero, self.qmin, self.qmax)
        kernel_lnvar = self.kernel_lnvar[...]

        if self.use_bias:
            b_mu = self.b_mu
            b_mu = fake_quant(b_mu, self.b_scale, self.b_zero, self.qmin, self.qmax)
            b_lnvar = self.b_lnvar            
        
        def _single_ex_forw(x, key):
            kernel_noise_key, b_key = jax.random.split(key)
            kernel_noise = jax.random.normal(kernel_noise_key, kernel_mu.shape)

            kernel = kernel_mu + kernel_noise * jnp.exp(0.5 * kernel_lnvar)

            x = x[jnp.newaxis, ...]  # [1, H, W, C_in]
            out = jax.lax.conv_general_dilated(
                lhs=x,
                rhs=kernel,
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

        return jax.vmap(_single_ex_forw, in_axes=(0, 0))(x, key_batch) # type: ignore
    