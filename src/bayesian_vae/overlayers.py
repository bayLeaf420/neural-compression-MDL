import jax
import jax.numpy as jnp
from flax import nnx


class QScale(nnx.Variable): pass
class QZeroPoint(nnx.Variable): pass
class OverParam(nnx.Param): pass


def fake_quant(x, scale, zero_point, qmin, qmax):
    """Asymmetric per-tensor fake-quant with STE. Scale/zero are treated as
    CONSTANTS w.r.t. gradients (correct QAT semantics — the quant grid is
    calibrated, not learned)."""
    scale = jax.lax.stop_gradient(scale)
    zero_point = jax.lax.stop_gradient(zero_point)
    q = jnp.clip(jnp.round(x / scale) + zero_point, qmin, qmax)
    x_hat = (q - zero_point) * scale
    return x + jax.lax.stop_gradient(x_hat - x)


def _calibrate_tensor(w, qmin, qmax):
    w_min, w_max = jnp.min(w), jnp.max(w)
    scale = jnp.maximum((w_max - w_min) / (qmax - qmin), 1e-8)
    zero = jnp.round(qmin - w_min / scale)
    return scale, zero


def _log1mexp(x):
    """Stable log(1 - exp(x)) for x <= 0."""
    return jnp.where(
        x > -jnp.log(2.0),
        jnp.log(-jnp.expm1(x)),
        jnp.log1p(-jnp.exp(x)),
    )


def weight_coding_nll(mean, lnvar, over_w, scale, zero_point, qmin, qmax):
    """Absolute bits (nats) to code the fake-quantized `over_w` under the frozen
    base distribution N(mean, exp(lnvar)), SUMMED over ALL weight elements.

    Works for ANY tensor shape (2D linear, 4D conv) — unlike the image-shaped
    erf_reconstruction_loss which hardcodes axis=(1,2,3) and a batch-mean.
    Uses the quant grid's true range as the boundary-bin edges, in weight units.
    """
    targ = fake_quant(over_w, scale, zero_point, qmin, qmax)

    lnvar = jnp.clip(lnvar, -25.0, 25.0)
    inv_std = jnp.exp(-0.5 * lnvar)
    half_bin = 0.5 * scale

    # boundary edges in WEIGHT units (not [0,1])
    low_edge = (qmin - zero_point) * scale
    high_edge = (qmax - zero_point) * scale

    lower = (targ - half_bin - mean) * inv_std
    upper = (targ + half_bin - mean) * inv_std

    # interior: stable log(Phi(upper) - Phi(lower))
    flip = (lower + upper) > 0.0
    a = jnp.where(flip, -upper, lower)
    b = jnp.where(flip, -lower, upper)
    log_cdf_a = jax.scipy.special.log_ndtr(a)
    log_cdf_b = jax.scipy.special.log_ndtr(b)
    log_prob_interior = log_cdf_b + _log1mexp(log_cdf_a - log_cdf_b)

    # boundary bins (mass to +/- inf)
    log_prob_low = jax.scipy.special.log_ndtr(upper)
    log_prob_high = jax.scipy.special.log_ndtr(-lower)

    at_low = targ <= low_edge + half_bin
    at_high = targ >= high_edge - half_bin
    log_prob = jnp.where(
        at_low, log_prob_low,
        jnp.where(at_high, log_prob_high, log_prob_interior),
    )
    # SUM over every element — absolute bit count for this whole tensor
    return -jnp.sum(log_prob)


class OverLin(nnx.Module):
    def __init__(self, base, bits: int = 8):
        self.in_dims = base.in_dims
        self.out_dims = base.out_dims
        self.use_bias = base.use_bias
        self.qmin = -(2 ** (bits - 1))
        self.qmax = (2 ** (bits - 1)) - 1

        self.over_w = OverParam(base.w_mu[...])
        self.w_mu_base = base.w_mu[...]
        self.w_lnvar_base = base.w_lnvar[...]
        self.w_scale = QScale(jnp.asarray(1.0))
        self.w_zero = QZeroPoint(jnp.asarray(0.0))

        if self.use_bias:
            self.over_b = OverParam(base.b_mu[...])
            self.b_mu_base = base.b_mu[...]
            self.b_lnvar_base = base.b_lnvar[...]
            self.b_scale = QScale(jnp.asarray(1.0))
            self.b_zero = QZeroPoint(jnp.asarray(0.0))

    def calibrate(self):
        s, z = _calibrate_tensor(self.over_w[...], self.qmin, self.qmax)
        self.w_scale[...], self.w_zero[...] = s, z
        if self.use_bias:
            s, z = _calibrate_tensor(self.over_b[...], self.qmin, self.qmax)
            self.b_scale[...], self.b_zero[...] = s, z

    def calculate_sampling_nll(self) -> jax.Array:
        nll = weight_coding_nll(
            self.w_mu_base, self.w_lnvar_base, self.over_w[...],
            self.w_scale[...], self.w_zero[...], self.qmin, self.qmax,
        )
        if self.use_bias:
            nll += weight_coding_nll(
                self.b_mu_base, self.b_lnvar_base, self.over_b[...],
                self.b_scale[...], self.b_zero[...], self.qmin, self.qmax,
            )
        return nll

    def __call__(self, x):
        w = fake_quant(self.over_w[...], self.w_scale[...], self.w_zero[...],
                       self.qmin, self.qmax)
        out = x @ w
        if self.use_bias:
            b = fake_quant(self.over_b[...], self.b_scale[...], self.b_zero[...],
                           self.qmin, self.qmax)
            out = out + b
        return out


class OverConv(nnx.Module):
    def __init__(self, base, bits: int = 8):
        self.use_bias = base.use_bias
        self.qmin = -(2 ** (bits - 1))
        self.qmax = (2 ** (bits - 1)) - 1

        self.strides = base.strides
        self.padding = base.padding
        self.input_dilation = base.input_dilation
        self.kernel_dilation = base.kernel_dilation
        self.feature_group_count = base.feature_group_count

        self.over_kernel = OverParam(base.kernel_mu[...])
        self.kernel_mu_base = base.kernel_mu[...]
        self.kernel_lnvar_base = base.kernel_lnvar[...]
        self.kernel_scale = QScale(jnp.asarray(1.0))
        self.kernel_zero = QZeroPoint(jnp.asarray(0.0))

        if self.use_bias:
            self.over_b = OverParam(base.b_mu[...])
            self.b_mu_base = base.b_mu[...]
            self.b_lnvar_base = base.b_lnvar[...]
            self.b_scale = QScale(jnp.asarray(1.0))
            self.b_zero = QZeroPoint(jnp.asarray(0.0))

    def calibrate(self):
        s, z = _calibrate_tensor(self.over_kernel[...], self.qmin, self.qmax)
        self.kernel_scale[...], self.kernel_zero[...] = s, z
        if self.use_bias:
            s, z = _calibrate_tensor(self.over_b[...], self.qmin, self.qmax)
            self.b_scale[...], self.b_zero[...] = s, z

    def calculate_sampling_nll(self) -> jax.Array:
        nll = weight_coding_nll(
            self.kernel_mu_base, self.kernel_lnvar_base, self.over_kernel[...],
            self.kernel_scale[...], self.kernel_zero[...], self.qmin, self.qmax,
        )
        if self.use_bias:
            nll += weight_coding_nll(
                self.b_mu_base, self.b_lnvar_base, self.over_b[...],
                self.b_scale[...], self.b_zero[...], self.qmin, self.qmax,
            )
        return nll

    def __call__(self, x):
        kernel = fake_quant(self.over_kernel[...], self.kernel_scale[...],
                            self.kernel_zero[...], self.qmin, self.qmax)
        out = jax.lax.conv_general_dilated(
            lhs=x, rhs=kernel,
            window_strides=self.strides, padding=self.padding,
            lhs_dilation=self.input_dilation, rhs_dilation=self.kernel_dilation,
            feature_group_count=self.feature_group_count,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        if self.use_bias:
            b = fake_quant(self.over_b[...], self.b_scale[...], self.b_zero[...],
                           self.qmin, self.qmax)
            out = out + b.reshape(1, 1, 1, -1)
        return out