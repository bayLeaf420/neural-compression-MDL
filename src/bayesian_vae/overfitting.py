import jax
import jax.numpy as jnp 
from flax import nnx
import optax
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

from bayesian_vae.train import (
    build_model,
    build_checkpoint_manager,
    save_if_best,
    run_validation,
    log_epoch,
)

from bayesian_vae.data_mnist import (
    load_mnist_train_images,
    load_mnist_test_images,
)

from bayesian_vae.losses import compute_validation_reconstruction_loss
from bayesian_vae.utils import PostLog, PriorParam
from bayesian_vae.models import PriorParam, BayesianVAE

NUM_STEPS = 400
MASTER_KEY = 99
BATCH_SIZE = 1 # At a time how many images to overfit on

def _compute_bits(
    model: BayesianVAE,
    x: jax.Array,
    params: jax.Array,
    key: jax.Array,
) -> jax.Array:
    x_hat = model(x, key)
    


@nnx.jit
def overfit_step(
        model: BayesianVAE,
        optimizer: nnx.Optimizer,
        input_batch: jax.Array,
        step_key: jaxArray,
        kl_weight_scale: jax.Array,
        decay: float,
) -> tuple[jax.Array, LossAux]:
    
    def loss_fn(model: BayesianVAE) -> jax.Array:
        loss = compute_validation_reconstruction_loss(
            model, input_batch, step_key,
        )
        return loss 
    
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grad = grad_fn(model)

    optimizer.update(model, grad)

    return loss

def main():
    key = jax.random.key(MASTER_KEY)
    key, model_key = jax.random.split(key)

    ### --- Get data --- ### 
    val_images = load_mnist_test_images() # [N, 28, 28, 1]
    batches =  len(val_images) // BATCH_SIZE
    kl_weight_scale = jnp.asarray(1.0)

    ### --- Model/ optimizer --- ### 
    model = build_model(model_key)
    abstract_state = nnx.state(model, (nnx.Param, PriorParam))

    manager = build_checkpoint_manager()
    step = manager.latest_step()
    restore = manager.restore(step, args=ocp.args.StandardRestore(abstract_state))




    optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)

    
