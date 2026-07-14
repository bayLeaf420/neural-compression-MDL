import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
import os
import numpy as np
import matplotlib.pyplot as plt

from bayesian_vae.newmodel import BayesVAE
from bayesian_vae.overmodel import OverVAE
from bayesian_vae.overlayers import OverParam
from bayesian_vae.overlosses import base_batch_loss, over_mdl_loss
from bayesian_vae.data_mnist import load_mnist_test_images


CHECKPOINT_DIR = os.path.abspath(os.environ.get("CHECKPOINT_DIR", "./checkpoints"))
MASTER_KEY = 123
BATCH_SIZE = 80
INNER_STEPS = 100
OVER_LR = 1e-3
BITS = 8
NUM_PIXELS = 28 * 28
RLN2 = 1.0 / jnp.log(2.0)
MAX_BATCHES = None            # set to an int (e.g. 20) for a pilot run


def load_trained_base(key):
    base = BayesVAE(rngs=nnx.Rngs(key))
    options = ocp.CheckpointManagerOptions(
        best_fn=lambda m: m["validation_loss"], best_mode="min"
    )
    mgr = ocp.CheckpointManager(CHECKPOINT_DIR, options=options)
    step = mgr.best_step()
    if step is None:
        step = mgr.latest_step()
    restored = mgr.restore(step, args=ocp.args.StandardRestore(nnx.state(base, nnx.Param)))
    nnx.update(base, restored)
    print(f"Restored base from step {step}")
    return base


@nnx.jit
def over_grad_step(over, optimizer, x):
    """Pure gradient step — calibration happens OUTSIDE (in the loop)."""
    def loss_fn(m):
        total, aux = over_mdl_loss(m, x)
        return total, aux
    (loss, aux), grad = nnx.value_and_grad(loss_fn, has_aux=True)(over)
    optimizer.update(over, grad)
    return loss, aux


def overfit_one_batch(base, x_batch):
    over = OverVAE(base, bits=BITS)
    optimizer = nnx.Optimizer(over, optax.adam(OVER_LR), wrt=OverParam)
    # --- one-time NaN diagnosis ---
    over.calibrate_all()
    for name, layer in zip(
        ["enc_c1","enc_c2","enc_lm","enc_ll","dec_l1","dec_c1","dec_cm","dec_cl"],
        over._layers()):
        nll = layer.calculate_sampling_nll()
        print(name, "sampling_nll:", float(nll),
              "scale:", float(getattr(layer, 'w_scale', getattr(layer,'kernel_scale'))[...]))
    # also check the forward pass separately
    xhm, xhl, zm, zl = over(x_batch)
    print("recon finite:", bool(jnp.all(jnp.isfinite(xhm))),
          "xhat_lnvar range:", float(jnp.min(xhl)), float(jnp.max(xhl)))
    # ---
    loss, aux = jnp.asarray(0.0), None
    for _ in range(INNER_STEPS):
        over.calibrate_all()                  # refresh quant grid OUTSIDE the grad step
        loss, aux = over_grad_step(over, optimizer, x_batch)
    return loss, aux


def main():
    key = jax.random.key(MASTER_KEY)
    key, base_key = jax.random.split(key)

    val_images = load_mnist_test_images()
    base = load_trained_base(base_key)

    n = val_images.shape[0]
    num_batches = n // BATCH_SIZE
    if MAX_BATCHES is not None:
        num_batches = min(num_batches, MAX_BATCHES)
    denom = BATCH_SIZE * NUM_PIXELS

    base_bpd_list, over_bpd_list, delta_list = [], [], []

    for bi in range(num_batches):
        x_batch = val_images[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]

        key, bkey = jax.random.split(key)
        base_nats = base_batch_loss(base, x_batch, bkey)
        base_bpd = float(base_nats * RLN2 / denom)

        over_nats, aux = overfit_one_batch(base, x_batch)
        over_bpd = float(over_nats * RLN2 / denom)

        base_bpd_list.append(base_bpd)
        over_bpd_list.append(over_bpd)
        delta_list.append(over_bpd - base_bpd)

        wpd = float(aux.weight_nll * RLN2 / denom) # type: ignore
        print(f"[batch {bi}/{num_batches}] base={base_bpd:.4f}  over={over_bpd:.4f}  "
              f"delta={over_bpd - base_bpd:+.4f}  (weight={wpd:.4f} bpd)")

    base_bpd = np.array(base_bpd_list)
    over_bpd = np.array(over_bpd_list)
    delta = np.array(delta_list)

    print(f"\nmean base bpd: {base_bpd.mean():.4f}")
    print(f"mean over bpd: {over_bpd.mean():.4f}")
    print(f"mean delta:    {delta.mean():+.4f}  ({'over WINS' if delta.mean() < 0 else 'base wins'})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(base_bpd, bins=30, alpha=0.6, label="base", color="steelblue")
    axes[0].hist(over_bpd, bins=30, alpha=0.6, label="overfit (MDL)", color="indianred")
    axes[0].set_xlabel("bits per dim"); axes[0].set_ylabel("count"); axes[0].legend()
    axes[0].set_title("Per-batch bpd: base vs overfit MDL")
    axes[1].hist(delta, bins=30, color="seagreen")
    axes[1].axvline(0.0, color="k", ls="--")
    axes[1].set_xlabel("over_bpd - base_bpd  (negative = overfit better)")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"Delta bpd (mean {delta.mean():+.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "overfit_batch_bpd_histogram.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()