import jax
import flax.linen as nn


class ResnetBlock(nn.Module):
    in_channels: int
    out_channels: int
    stride: int = 1
    expansion: int = 1

    @nn.compact
    def __call__(
        self, x: jax.Array, train: bool, activation: callable = nn.relu
    ) -> jax.Array:
        residual = x

        if self.stride != 1 or self.out_channels != self.in_channels * self.expansion:
            residual = nn.Conv(
                self.out_channels * self.expansion,
                strides=self.stride,
                kernel_size=1,
                use_bias=False,
            )(residual)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        x = nn.Conv(
            self.out_channels,
            kernel_size=3,
            strides=self.stride,
            padding=1,
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = activation(x)

        x = nn.Conv(
            self.out_channels, kernel_size=3, strides=1, padding=1, use_bias=False
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = x + residual
        x = activation(x)

        return x
