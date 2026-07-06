from dataclasses import dataclass


@dataclass(frozen=True)
class ConvConfig:
    kernels: tuple[tuple[int, int], ...]
    strides: tuple[tuple[int, int], ...]
    channels: tuple[int, ...]


@dataclass(frozen=True)
class LinConfig:
    hidden_dims: tuple[int, ...]


@dataclass(frozen=True)
class EncoderConfig:
    conv: ConvConfig
    lin: LinConfig


@dataclass(frozen=True)
class DecoderConfig:
    lin: LinConfig


@dataclass(frozen=True)
class VaeConfig:
    encoder_config: EncoderConfig
    decoder_config: DecoderConfig
    z_dim: int
    z_free_nats: float
    w_prior_lnvar: float
   