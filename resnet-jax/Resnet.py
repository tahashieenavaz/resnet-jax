import jax
import flax.linen as nn
from typing import Callable


class Resnet(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        train: bool = False,
        activation: Callable[[jax.Array], jax.Array] = nn.relu,
    ) -> jax.Array:
        x = nn.Conv(64, kernel_size=(7, 7), strides=2, padding=3, use_bias=False)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = activation(x)
        x = nn.max_pool(window_shape=(3, 3), strides=2, padding="SAME")

        # four layers go here

        x = jax.numpy.mean(x, axis=(1, 2), keepdims=True)
        return nn.Dense(self.num_classes)
