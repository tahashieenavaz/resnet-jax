import jax
import flax.linen as nn
from .blocks import ResnetEntryBlock
from typing import Callable


class Resnet(nn.Module):
    num_classes: int
    activation: Callable[[jax.Array], jax.Array] = nn.relu

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        train: bool = False,
    ) -> jax.Array:
        x = ResnetEntryBlock()(x, train=train)

        # four layers go here

        x = jax.numpy.mean(x, axis=(1, 2), keepdims=True)
        return nn.Dense(self.num_classes)
