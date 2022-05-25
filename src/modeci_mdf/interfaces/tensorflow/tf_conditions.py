from abc import abstractmethod
from typing import Dict
import tensorflow as tf

from modeci_mdf.interfaces.tensorflow.classes import TFNodeContext


class TFCondition(tf.Module):
    def __init__(self, name: str):
        super(TFCondition, self).__init__(name=name)

    @abstractmethod
    def __call__(self, **kwargs):
        pass


class Always(TFCondition):
    def __init__(self, condition: Dict):
        super(Always, self).__init__(name="Always")

    @tf.function
    def __call__(self, graph_ctx: Dict[str, TFNodeContext]):
        return tf.constant(True)


class EveryNCalls(TFCondition):
    def __init__(self, condition: Dict):
        super(EveryNCalls, self).__init__("EveryNCalls")
        self.dependency = condition["args"]["dependencies"]
        self.n = condition["args"]["n"]

    @tf.function
    def __call__(self, graph_ctx: Dict[str, TFNodeContext]):
        dependent_ctx = graph_ctx[self.dependency]
        return dependent_ctx.metadata.num_executions % self.n == 0


class BeforeNCalls(TFCondition):
    def __init__(self, condition: Dict):
        super(BeforeNCalls, self).__init__("BeforeNCalls")
        self.dependency = condition["args"]["dependencies"]
        self.n = condition["args"]["n"]

    @tf.function
    def __call__(self, graph_ctx: Dict[str, TFNodeContext]):
        dependent_ctx = graph_ctx[self.dependency]
        return dependent_ctx.metadata.num_executions < self.n


class AfterNCalls(TFCondition):
    def __init__(self, condition: Dict):
        super(AfterNCalls, self).__init__("AfterNCalls")
        self.dependency = condition["args"]["dependencies"]
        self.n = condition["args"]["n"]

    @tf.function
    def __call__(self, graph_ctx: Dict[str, TFNodeContext]):
        dependent_ctx = graph_ctx[self.dependency]
        return dependent_ctx.metadata.num_executions > self.n


condition_map = {
    "Always": Always,
    "EveryNCalls": EveryNCalls,
    "BeforeNCalls": BeforeNCalls,
    "AfterNCalls": AfterNCalls,
}


def create_condition(name: str, spec: Dict):
    if name not in condition_map:
        raise Exception(
            f"Tensorflow interface currently doesn't support MDF condition {name}!"
        )
    return condition_map[name](spec)
