import jax
import flax.linen as nn


class ResnetBasicBlock(nn.Module):
    in_channels: int
    out_channels: int
    stride: int = 1
    expansion: int = 1

    def should_project(self):
        return (
            self.stride != 1 or self.out_channels != self.in_channels * self.expansion
        )

    @nn.compact
    def __call__(
        self, x: jax.Array, train: bool, activation: callable = nn.relu
    ) -> jax.Array:
        residual = x

        if self.should_project():
            residual = nn.Conv(
                self.out_channels * self.expansion,
                strides=self.stride,
                kernel_size=1,
                use_bias=False,
                name="projection_convolutional",
            )(residual)
            residual = nn.BatchNorm(
                use_running_average=not train, name="projection_batch_norm"
            )(residual)

        x = nn.Conv(
            self.out_channels,
            kernel_size=3,
            strides=self.stride,
            use_bias=False,
            name="first_convolutional",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="first_batch_norm")(x)
        x = activation(x)

        x = nn.Conv(
            self.out_channels,
            kernel_size=3,
            strides=1,
            use_bias=False,
            name="second_convolutional",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="second_batch_norm")(x)
        x = x + residual
        x = activation(x)

        return x
