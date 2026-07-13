import jax
from flax import nnx
import jax.numpy as jnp
import qwix 

from bayesian_vae.layers import BayesianLinear, BayesianConv2D 
from bayesian_vae.losses import erf_reconstruction_loss 

class QScale(nnx.Variable): pass
class QZeroPoint(nnx.Variable): pass 
class OverParam(nnx.Param): pass # overfitting w.r.t. OverParam


def fake_quant(input, scale, zero_point, qmin, qmax):
    """Asymmetric per-tensor fake-quant with straight-through estimator."""
    q = jnp.clip(jnp.round(input / scale) + zero_point, qmin, qmax)
    input_hat = (q - zero_point) * scale
    return input + jax.lax.stop_gradient(input_hat - input)   # <-- STE: grad flows as identity


def _scaled_erf(mean, lnvar, over_w, scale, zero_point, qmin, qmax):
    targ = fake_quant(over_w, scale, zero_point, qmin, qmax)
    return erf_reconstruction_loss(mean, lnvar, targ, bin_width=scale)


class OverLin(BayesianLinear):
    """Uses mean of weight distribution for forward pass only.
    """
    def __init__(self, in_dims, out_dims, bits=8, *, rngs: nnx.Rngs):
        """
        Initialize overfitter layer with QAT.
        """
        super().__init__(in_dims, out_dims, rngs=rngs)
        self.over_w = OverParam(self.w_mu[...]) # overfitting w.r.t. OverParam
        self.qmin = -(2 ** (bits - 1)) # -128
        self.qmax = (2 ** (bits-1)) - 1 # 127
        # per tensor: scalar scale + scalar zero point for whole tensor
        self.w_scale = QScale(jnp.asarray(1.0))
        self.w_zero = QScale(jnp.asarray(0.0))
        if self.use_bias:
            self.over_b = OverParam(self.b_mu[...]) # type: ignore
            self.b_scale = QScale(jnp.asarray(1.0))
            self.b_zero = QScale(jnp.asarray(0.0))

        
    def calibrate(self):
        """
        Refresh per-tensor scale/zero from current weights. 
        Call each step.
        """
        w = self.over_w[...]
        w_min, w_max = jnp.min(w), jnp.max(w)
        w_scale = jnp.maximum((w_max - w_min) / (self.qmax - self.qmin), 1e-8)
        self.w_scale[...] = w_scale
        self.w_zero[...] = jnp.round(self.qmin - w_min / w_scale)

        if self.use_bias:
            b = self.over_b[...] # type: ignore
            b_min, b_max = jnp.min(b), jnp.max(b)
            b_scale = jnp.maximum((b_max - b_min) / (self.qmax - self.qmin), 1e-8)
            self.b_scale[...] = b_scale 
            self.b_zero[...] = jnp.round(self.qmin - b_min / b_scale)
    
    def calculate_sampling_nll(self):
        """Calculate reconstruction loss based on initial distribution"""
        prior_mu = self.w_mu[...]
        prior_lnvar = self.w_lnvar[...]

        
    def __call__(
        self,
        x: jax.Array, # [B, d_in]
    ) -> jax.Array:
        """
        Overfitter layer custom forward pass. Take the mean to avoid 
        having to do 
        """
        w = self.over_w[...] # [d_in, d_out]
        w = fake_quant(w, self.w_scale, self.w_zero, self.qmin, self.qmax)
        out = x @ w # out is [B, d_out]
        if self.use_bias:
            b = self.over_b[...] # type: ignore
            b = fake_quant(b, self.b_scale, self.b_zero, self.qmin, self.qmax)
            out += b # type: ignore
        return out 
    
        
class OverConv(BayesianConv2D):
    def __init__(self, in_channels, out_channels, kernel_size, strides: tuple[int, int] | None=None, bits=8, *, rngs: nnx.Rngs):
        super().__init__(in_channels, out_channels, kernel_size, strides=strides, rngs=rngs) # type: ignore
        self.over_kernel = OverParam(self.kernel_mu[...])
        self.qmin = -(2 ** (bits - 1)) # -128
        self.qmax = (2 ** (bits-1)) - 1 # 127
        self.kernel_scale = QScale(jnp.asarray(1.0))
        self.kernel_zero = QScale(jnp.asarray(0.0))
        if self.use_bias:
            self.over_b = OverParam(self.b_mu)
            self.b_scale = QScale(jnp.asarray(1.0))
            self.b_zero = QScale(jnp.asarray(0.0))
    
    def calibrate(self):
        """
        Refresh per-tensor scale/zero from current weights. 
        Call each step.
        """
        kernel = self.over_kernel[...]
        kernel_min, kernel_max = jnp.min(kernel), jnp.max(kernel)
        kernel_scale = jnp.maximum((kernel_max - kernel_min) / (self.qmax - self.qmin), 1e-8)
        self.kernel_scale[...] = kernel_scale
        self.kernel_zero[...] = jnp.round(self.qmin - kernel_min / kernel_scale)

        if self.use_bias:
            b = self.over_b[...] # type: ignore
            b_min, b_max = jnp.min(b), jnp.max(b)
            b_scale = jnp.maximum((b_max - b_min) / (self.qmax - self.qmin), 1e-8)
            self.b_scale[...] = b_scale 
            self.b_zero[...] = jnp.round(self.qmin - b_min / b_scale)


    def __call__(self, x: jax.Array) -> jax.Array:
        kernel = self.over_kernel[...]
        kernel = fake_quant(kernel, self.kernel_scale, self.kernel_zero, self.qmin, self.qmax)

        out = jax.lax.conv_general_dilated(
            lhs=x, # [B, H, W, Cin]
            rhs=kernel, # [kH, kW, Cin, Cout]
            window_strides=self.strides,
            padding=self.padding,
            lhs_dilation=self.input_dilation,
            rhs_dilation=self.kernel_dilation,
            feature_group_count=self.feature_group_count,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )

        if self.use_bias:
            b = self.over_b[...] # type: ignore
            b = fake_quant(b, self.b_scale, self.b_zero, self.qmin, self.qmax)
            out = out + b.reshape(1, 1, 1, -1) # bias has shape [Cout, ]

        return out 
    



    

        


        
