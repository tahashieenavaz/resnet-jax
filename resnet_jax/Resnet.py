import jax
import flax.linen as nn
from .blocks import ResnetEntryBlock
from typing import Callable


class Resnet(nn.Module):
    num_classes: int
    activation: Callable[[jax.Array], jax.Array] = nn.relu

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
        x = ResnetEntryBlock()(x, train=train)
        # four layers go here
        x = jax.numpy.mean(x, axis=(1, 2))
        return nn.Dense(self.num_classes)(x)


if __name__ == "__main__":
    model = Resnet(num_classes=10)
    key = jax.random.PRNGKey(0)
    key_data, key_init = jax.random.split(key)
    dummy_input = jax.random.normal(key_data, (1, 224, 224, 3))
    output, variables = model.init_with_output(key_init, dummy_input)
    print("Output shape:", output.shape)
