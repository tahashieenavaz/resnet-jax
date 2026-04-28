import jax
import flax.linen as nn
from typing import Callable


class ResnetEntryBlock(nn.Module):
    activation: Callable[[jax.Array], jax.Array] = nn.relu

    @nn.compact
    def __call__(self, x: jax.Array, *, train: bool = False) -> jax.Array:
        x = nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding=3, use_bias=False)(
            x
        )
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        return x
