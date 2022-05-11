import sys
from typing import List
from pyrsistent import b
import tensorflow as tf

from modeci_mdf.interfaces.tensorflow.exporter import TFNode


@tf.function(experimental_follow_type_hints=True, jit_compile=True)
def Always(*args, **kwargs) -> tf.Tensor:
    return True


@tf.function(experimental_follow_type_hints=True, jit_compile=True)
def EveryNCalls(nodes: List[TFNode], threshold: int, *args, **kwargs) -> tf.Tensor:
    return nodes[0].metadata["execution_count"] % threshold == 0

@tf.function(experimental_follow_type_hints=True, jit_compile=True)
def BeforeNCalls(nodes: List[TFNode], threshold: int, *args, **kwargs) -> tf.Tensor:
    return nodes[0].metadata["execution_count"] % threshold == 0

condition_map = {"Always": Always}

def get_condition(name: str):
    if name not in condition_map:
        raise Exception(
            f"Tensorflow interface currently doesn't support MDF condition {name}!"
        )
    return condition_map[name]
