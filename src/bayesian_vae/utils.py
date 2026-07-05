import jax
from flax import nnx, struct


class PriorParam(nnx.Variable):
    """Custom class for prior learning to be slowed down, so that posterior-prior collapse
    is less likely in the Bits-Back bits as loss scheme.
    """
    pass

@struct.dataclass
class PostLog:
    """Class to log posterior mean and variances over a batch"""
    z_mu: jax.Array # [B, z_dim]
    z_lnvar: jax.Array # [B, z_sim]
