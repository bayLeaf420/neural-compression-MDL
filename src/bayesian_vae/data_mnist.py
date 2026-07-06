import gzip
import os
import struct
import numpy as np
import jax
import jax.numpy as jnp

FILE_DIR = os.path.abspath(os.environ.get("FILE_DIR", "./data"))
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
}


def _read_idx_images(path: str) -> jax.Array:
    """Parse an IDX3 image file (gzipped) intoa  (N, H, W) uint8 array.

    Args:
        path: Contains file name and directory
    """
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic number {magic} in image file {path}")
        buf = f.read(num * rows * cols)
        data = jnp.asarray(np.frombuffer(buf, dtype=jnp.uint8))
        return data.reshape(num, rows, cols)


def _load_images(filename: str) -> jax.Array:
    """Download, parse, and normalise to [0, 1], add channel fim
    -> (N, 28, 28, 1)
    """
    path = os.path.join(FILE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"MNIST file not found: {path}. Place the .gz files in {FILE_DIR}.")
    images = _read_idx_images(path).astype(jnp.float32) / 255.0
    images = images[..., jnp.newaxis]  # [N, 28, 28] -> [N, 28, 28, 1]
    return images


def load_mnist_train_images() -> jax.Array:
    return _load_images(_FILES["train_images"])


def load_mnist_test_images() -> jax.Array:
    return _load_images(_FILES["test_images"])


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
