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


if __name__ == "__main__":
    model = Resnet(num_classes=10)
    key = jax.random.PRNGKey(0)
    data_key, init_key = jax.random.split(key)
    dummy_input = jax.random.normal(data_key, (1, 224, 224, 3))
    out, variables = model.init_with_output(init_key, dummy_input)
    print("Output shape:", out.shape)
