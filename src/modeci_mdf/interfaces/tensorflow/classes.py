from typing import Dict, NamedTuple
import tensorflow as tf


class TFNodeMetadata(NamedTuple):
    num_executions: tf.Tensor

    @staticmethod
    @tf.function
    def step_numexecutions(metadata):
        return TFNodeMetadata(num_executions=metadata.num_executions + 1)


class TFNodeContext(NamedTuple):
    parameter_ctx: Dict[str, tf.Tensor]
    input_port_ctx: Dict[str, tf.Tensor]
    metadata: TFNodeMetadata
