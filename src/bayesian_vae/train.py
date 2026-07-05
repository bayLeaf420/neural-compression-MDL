import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
import os
import matplotlib.pyplot as plt

from .models import BayesianVAE
from .losses import (
    compute_training_loss,
    compute_validation_reconstruction_loss,
    LossAux,
)
from .config import EncoderConfig, DecoderConfig, ConvConfig, LinConfig, VaeConfig
from .utils import PostLog, PriorParam

from .data_mnist import (
    load_mnist_train_images,
    load_mnist_test_images,
    iterate_shuffled_batches,
    validation_iterator,
)

CHECKPOINT_DIR = os.path.abspath("./checkpoints")
NUM_EPOCHS = 50
BATCH_SIZE = 128
VALIDATE_EVERY = 5
MASTER_KEY = 47

def prior_train_step_ema(
        z_prior_mu: jax.Array, z_prior_lnvar: jax.Array,
        z_mu: jax.Array, z_lnvar: jax.Array, 
        decay: float,
        ) -> tuple[jax.Array, jax.Array]:
    """Seperate update rule for PriorParam. We run this after running

    Args: 
        z_prior_mu: Current prior mean
        z_prior_lnvar: Current prior natural log-variance
        z_mu: Posterior mean of latent
        z_lnvar: Posterior lnvar of latent
        decay: -->1 ==> Slower updates to prior.
    """
    # aggregate posterior over the batch
    batch_mu = jnp.mean(z_mu, axis=0)                    # [z_dim]
    batch_var = jnp.mean(jnp.exp(z_lnvar) + z_mu**2, axis=0) - batch_mu**2
    
    new_z_mu = decay * z_prior_mu + (1 - decay) * batch_mu

    new_z_var = decay * jnp.exp(z_prior_lnvar) + (1 - decay) * batch_var
    new_z_var = jnp.maximum(new_z_var, 1e-6)   # guarantee positivity, JIT-safe
    new_z_lnvar = jnp.log(new_z_var)

    return new_z_mu, new_z_lnvar


@nnx.jit(static_argnames=('decay',))  # JIT compile this thing
def train_step(
    model: BayesianVAE,
    optimizer: nnx.Optimizer,
    input_batch: jax.Array,
    step_key: jax.Array,
    kl_weight_scale: jax.Array,
    decay: float,
) -> tuple[jax.Array, LossAux, PostLog]:
    """
    """

    def loss_fn(model: BayesianVAE) -> tuple[jax.Array, tuple[LossAux, PostLog]]:
        loss, aux, post_log = compute_training_loss(model, input_batch, step_key, kl_weight_scale)
        return loss, (aux, post_log)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (aux, post_log)), grad = grad_fn(model)

    # Update 'model''s params and 'optimizer''s opt_state in place
    optimizer.update(model, grad)

    # Update model's prior
    model.z_prior_mu[...], model.z_prior_lnvar[...] = prior_train_step_ema(
        model.z_prior_mu[...], 
        model.z_prior_lnvar[...],
        post_log.z_mu,
        post_log.z_lnvar,
        decay,
    )
    return loss, aux, post_log


@nnx.jit
def validation_step(
    model: BayesianVAE,
    input_batch: jax.Array,
    step_key: jax.Array,
) -> jax.Array:
    return compute_validation_reconstruction_loss(model, input_batch, step_key)


def build_model(key: jax.Array) -> BayesianVAE:
    encoder_config = EncoderConfig(
        ConvConfig(
            kernels=((3, 3), (3, 3), (3, 3)),
            strides=((1, 1), (1, 1), (1, 1)),
            channels=(7, 7, 7),
        ),
        LinConfig(
            hidden_dims=(638, 492, 394, 296, 197, 98, 45),
        ),
    )
    decoder_config = DecoderConfig(
        LinConfig(
            hidden_dims=(98, 148, 197, 247, 296, 345, 394, 492, 638, 711, 784),
        )
    )
    vae_config = VaeConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        z_dim = 45,
        w_prior_lnvar=0.0,
    )

    return BayesianVAE(
        in_shape=(28, 28, 1),
        config=vae_config,
        rngs=nnx.Rngs(key),
    )


def build_checkpoint_manager() -> ocp.CheckpointManager:
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        best_fn=lambda metrics: metrics["validation_loss"],
        best_mode="min",
    )

    return ocp.CheckpointManager(CHECKPOINT_DIR, options=options)


def save_if_best(
    manager: ocp.CheckpointManager,
    model: BayesianVAE,
    epoch: int,
    val_loss: jax.Array,
) -> None:
    """Save only the Param state, avoiding key<fry> dtype.

    Args:

    """
    params = nnx.state(model, (nnx.Param, PriorParam))
    manager.save(
        epoch,
        args=ocp.args.StandardSave(params),
        # Orbax needs validation loss as a float, but it's passed as a jax.Array
        metrics={"validation_loss": float(val_loss)}, 
    )
    print(f"Saved checkpoint at epoch {epoch} (val_loss={val_loss:.4f})")


def run_validation(
    model: BayesianVAE,
    validation_images: jax.Array,
    batch_size: int,
    master_key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Returns (avg_val_loss, updated_master_key)"""
    total_val_loss = jnp.asarray(0.0)
    batches = validation_images.shape[0] // batch_size
    for val_batch in validation_iterator(validation_images, batch_size, batches):
        master_key, step_key = jax.random.split(master_key)
        total_val_loss += validation_step(model, val_batch, step_key)
    avg_val_loss = total_val_loss / float(batches)
    return avg_val_loss, master_key


def train_one_epoch(
    model: BayesianVAE,
    optimizer: nnx.Optimizer,
    train_images: jax.Array,
    kl_weight_scale: jax.Array,
    batch_size: int,
    epoch: int,
    key: jax.Array,
) -> tuple[jax.Array, LossAux, jax.Array]:
    """Returns (last_loss, last_aux, updated_key)."""
    key, shuffle_key = jax.random.split(key)  # Pass 'key' as master_key
    loss = jnp.asarray(0.0)
    aux = None
    batches = train_images.shape[0] // batch_size
    decay = float(1 - 1 / batches)
    
    for image_batch in iterate_shuffled_batches(
        train_images, shuffle_key, batch_size, batches
    ):
        key, step_key = jax.random.split(key)
        loss, aux, post_log = train_step(
            model,
            optimizer,
            image_batch,
            step_key,
            kl_weight_scale,
            decay,
        )
        if float(jnp.max(post_log.z_lnvar)) > 15.0:   # Detect unstable training
            print(f"WARNING: z_lnvar max ... at epoch {epoch}")
    assert aux is not None  # at least one batch ran
    return loss, aux, key


def log_epoch(
    epoch: int,
    loss: jax.Array,
    aux: LossAux,
    avg_val_loss: float | jax.Array,
) -> None:
    
    print(
        f"epoch {epoch}: total_loss={loss:.4f}"
        f" reconstruction_loss={aux.reconstruction_loss:.4f}"
        f" latent_kl_loss={aux.latent_kl_divergence:.4f}"
        f" weight_kl_loss={aux.weight_kl_divergence:.4f}"
        f" | val_loss={avg_val_loss:.4f}"
    )


def main() -> None:
    key = jax.random.key(MASTER_KEY)
    key, model_key = jax.random.split(key)

    ### ---- Get data ---- ###
    train_images = load_mnist_train_images()
    val_images = load_mnist_test_images()
    batches = len(train_images) // BATCH_SIZE
    kl_weight_scale = jnp.asarray(1.0 / batches)

    ### ---- Model / optimizer ---- ###
    model = build_model(model_key)
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)

    ### ---- Checkpointing ---- ###
    manager = build_checkpoint_manager()
    best_val_loss = float("inf")

    train_loss_array = []
    val_loss_array = []
    
    ### ---- Training loop ---- ###
    for epoch in range(NUM_EPOCHS):
        loss, aux, key = train_one_epoch(
            model,
            optimizer,
            train_images,
            kl_weight_scale,
            BATCH_SIZE,
            epoch,
            key,
        )
        avg_val_loss = float("nan")
        is_val_epoch = epoch % VALIDATE_EVERY == 0 or epoch == NUM_EPOCHS - 1
        if is_val_epoch:
            avg_val_loss, key = run_validation(
                model,
                val_images,
                BATCH_SIZE,
                key,
            )
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_if_best(manager, model, epoch, avg_val_loss)

        log_epoch(epoch, loss, aux, avg_val_loss)
        train_loss_array.append(loss)
        val_loss_array.append(avg_val_loss)

    # Orbax saving is asynchronous, we main() to wait for it to finish saving before returning.
    manager.wait_until_finished() 

    # ---- Plot training graphs ----
    epoch_range = jnp.arange(0, NUM_EPOCHS)
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    axes[0].plot(epoch_range, jnp.asarray(train_loss_array))
    axes[0].set_xlabel('epochs')
    axes[0].set_ylabel('Training loss (Nats)')
    axes[1].plot(epoch_range, jnp.asarray(val_loss_array))
    axes[1].set_xlabel('epochs')
    axes[1].set_ylabel('Validation loss (Nats)')
    plt.tight_layout()
    plt.show()
    

if __name__=="__main__":
    main()