import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

import os 

from bayesian_vae.train import (
    BATCH_SIZE,
    VALIDATE_EVERY,
    build_model,
    save_if_best,
    run_validation,
    train_one_epoch,
    log_epoch,
)

from bayesian_vae.data_mnist import (
    load_mnist_train_images,
    load_mnist_test_images,
)

from bayesian_vae.models import PriorParam

# Redefine NUM_EPOCHS and MASTER_KEY
NUM_EPOCHS = 100
MASTER_KEY = 78
CHECKPOINT_DIR = os.path.abspath(os.environ.get("CHECKPOINT_DIR_CONT", "./checkpoints"))


def build_checkpoint_manager() -> ocp.CheckpointManager:
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        best_fn=lambda metrics: metrics["validation_loss"],
        best_mode="min",
    )

    return ocp.CheckpointManager(CHECKPOINT_DIR, options=options)


def main() -> None:
    key = jax.random.key(MASTER_KEY)

    ### ---- Get data ---- ###
    train_images = load_mnist_train_images()
    val_images = load_mnist_test_images()
    batches = len(train_images) // BATCH_SIZE
    kl_weight_scale = jnp.asarray(1.0 / batches)

    ### ---- Load Saved Model ---- ###
    model = build_model(jax.random.key(0))
    abstract_state = nnx.state(model, (nnx.Param, PriorParam))
    
    manager = build_checkpoint_manager()
    step = manager.latest_step()
    restored = manager.restore(step, args=ocp.args.StandardRestore(abstract_state))

    nnx.update(model, restored)

    ### ---- Get optimizer and continue ---- ###
    best_val_loss = float("inf")
    optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)

    train_loss_array = []
    val_loss_array = []
    ln2 = jnp.log(2)
    num_pix = float(28*28)
    
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
        val_loss_array.append(avg_val_loss/(ln2 * num_pix))

    # Orbax saving is asynchronous, we main() to wait for it to finish saving before returning.
    manager.wait_until_finished() 

    # ---- Plot training graphs ----
    val_loss_array = jnp.asarray(val_loss_array)
    clean_val = jnp.where(jnp.isnan(val_loss_array), 0.0, val_loss_array)
    epoch_range = jnp.arange(0, NUM_EPOCHS)
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    axes[0].plot(epoch_range, jnp.asarray(train_loss_array))
    axes[0].set_xlabel('epochs')
    axes[0].set_ylabel('Training loss (Nats)')
    axes[1].plot(epoch_range, clean_val)
    axes[1].set_xlabel('epochs')
    axes[1].set_ylabel('Validation loss (Bits per dim)')
    plt.tight_layout()
    plt.show()
    

if __name__=="__main__":
    main()