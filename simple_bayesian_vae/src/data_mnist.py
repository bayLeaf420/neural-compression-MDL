from torchvision import datasets
import numpy as np
import jax
import jax.numpy as jnp


def load_mnist_train_images() -> jax.Array:
    dataset = datasets.MNIST(root="./data", train=True, download=True)
    images = dataset.data.numpy().astype(np.float32) / 255.0
    images = jnp.asarray(images)
    images = images[..., jnp.newaxis]  # Add channel dim -> (N, 28, 28, 1)
    return images


def load_mnist_test_images() -> jax.Array:
    dataset = datasets.MNIST(root="./data", train=False, download=True)
    images = dataset.data.numpy().astype(np.float32) / 255.0
    images = jnp.asarray(images)
    images = images[..., jnp.newaxis]
    return images


def iterate_shuffled_batches(
    images: jax.Array,
    shuffle_key: jax.Array,
    batch_size: int,
    num_batches: int,
):
    permutation = jax.random.permutation(shuffle_key, images.shape[0])
    shuffled_images = images[permutation]
    for batch_index in range(num_batches):
        start = batch_index * batch_size
        end = start + batch_size
        yield shuffled_images[start:end]


def validation_iterator(images: jax.Array, batch_size: int, num_batches: int):
    # No iterator needed
    for batch_index in range(num_batches):
        start = batch_index * batch_size
        end = start + batch_size
        yield images[start:end]
