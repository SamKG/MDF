"""
Functions for converting from MDF models to PyTorch models.
"""
from typing import Callable, Dict, Set, Tuple, Union

import tensorflow as tf

from modeci_mdf.execution_engine import EvaluableGraph
from modeci_mdf.interfaces.tensorflow.builtins import get_builtin
from modeci_mdf.interfaces.tensorflow.classes import TFNodeContext, TFNodeMetadata
from modeci_mdf.interfaces.tensorflow.tf_conditions import create_condition
from modeci_mdf.mdf import Graph, Model, Node, Parameter


def is_function_param(parameter: Parameter) -> bool:
    return getattr(parameter, "function") is not None


def parameter_to_tensorflow(parameter: Parameter) -> tf.Operation:
    if (function := getattr(parameter, "function")) is not None:
        return get_builtin(function)
    elif (value := getattr(parameter, "value")) is not None:
        return tf.constant(value, name=parameter.get_id(), dtype=tf.double)
    else:
        raise Exception(f"Don't know how to convert parameter {parameter}!")


class TFNode(tf.Module):
    def generate_node_context(self) -> TFNodeContext:
        return TFNodeContext(
            parameter_ctx=dict((p, t) for (p, t) in self.nonfunc_params.items()),
            input_port_ctx=dict(
                (p, tf.constant(0.0, dtype=tf.double, name=p))
                for p in self.input_port_ids
            ),
            metadata=TFNodeMetadata(num_executions=tf.constant(0, dtype=tf.int32)),
        )

    def __init__(self, node: Node):
        super(TFNode, self).__init__(name=node.get_id())

        self.func_params: Dict[str, Callable] = {
            p.get_id(): parameter_to_tensorflow(p)
            for p in node.parameters
            if is_function_param(p)
        }
        self.nonfunc_params: Dict[str, tf.Tensor] = {
            p.get_id(): parameter_to_tensorflow(p)
            for p in node.parameters
            if not is_function_param(p)
        }

        self.input_port_ids: Set[str] = set(p.get_id() for p in node.input_ports)

        # map from node params/ports to function parameters
        self.func_args: Dict[str, Dict[str, str]] = {}
        for f in [f for f in node.parameters if is_function_param(f)]:
            f_args = {}
            for (arg, val) in f.args.items():
                f_args[val] = arg
            self.func_args[f.get_id()] = f_args

        self.output_port_mappings: Dict[str, str] = {
            o.get_id(): o.value for o in node.output_ports
        }

    @tf.function
    def __call__(self, node_ctx: TFNodeContext):
        func_outputs = {}
        for (func_id, function) in self.func_params.items():
            func_args = {}
            for (val, argname) in self.func_args[func_id].items():
                if val in node_ctx.parameter_ctx:
                    func_args[argname] = node_ctx.parameter_ctx[val]
                elif val in node_ctx.input_port_ctx:
                    func_args[argname] = node_ctx.input_port_ctx[val]
                else:
                    raise Exception(f"Couldn't find {val} in node_ctx!")
            func_outputs[func_id] = function(**func_args)

        output_ports: Dict[str, tf.Tensor] = {}
        for (o, v) in self.output_port_mappings.items():
            if v in func_outputs:
                output_ports[o] = func_outputs[v]
            elif v in node_ctx.parameter_ctx:
                output_ports[o] = node_ctx.parameter_ctx[v]
            elif v in node_ctx.input_port_ctx:
                output_ports[o] = node_ctx.input_port_ctx[v]

        updated_metadata = TFNodeMetadata.step_numexecutions(node_ctx.metadata)
        node_ctx = TFNodeContext(
            metadata=updated_metadata,
            parameter_ctx=node_ctx.parameter_ctx,
            input_port_ctx=node_ctx.input_port_ctx,
        )

        return output_ports, node_ctx


def node_to_tensorflow(node: Node) -> Union[Callable, tf.Variable]:
    return TFNode(node)


class TFGraph(tf.Module):
    def generate_node_contexts(self) -> Dict[str, TFNodeContext]:
        return {s: n.generate_node_context() for (s, n) in self.graph_nodes.items()}

    def __init__(self, graph: Graph):
        super(TFGraph, self).__init__(name=graph.get_id())
        scheduler = EvaluableGraph(graph=graph).scheduler
        self.consideration_queue = [
            set(n.get_id() for n in s) for s in scheduler.consideration_queue
        ]

        self.graph_nodes: Dict[str, TFNode] = {}
        for n in graph.nodes:
            self.graph_nodes[n.get_id()] = node_to_tensorflow(n)

        self.node_conditions = {
            node_id: create_condition(spec["type"], spec)
            for (node_id, spec) in graph.conditions.node_specific.items()
        }

        self.termination_condition = create_condition(
            graph.conditions.termination["environment_state_update"]["type"],
            graph.conditions.termination["environment_state_update"],
        )

        self.graph_edges: Dict[str, Tuple[str, str, str]] = {
            n.get_id(): [] for n in graph.nodes
        }
        for edge in graph.edges:
            self.graph_edges[edge.sender].append(
                (edge.sender_port, edge.receiver, edge.receiver_port)
            )

    @tf.function
    def __call__(
        self,
        node_ctxs: Dict[str, TFNodeContext],
    ):
        node_outputs = {}
        node_ctxs = node_ctxs.copy()

        while not self.termination_condition(node_ctxs):
            active_nodes = {k: True for k in self.graph_nodes.keys()}

            for exec_set in self.consideration_queue:
                for node_id in exec_set:
                    if node_id in self.node_conditions and not self.node_conditions[
                        node_id
                    ](node_ctxs):
                        active_nodes[node_id] = False

                    if not active_nodes[node_id]:
                        continue
                    outputs, node_ctx = self.graph_nodes[node_id](node_ctxs[node_id])

                    node_outputs = {
                        k: v if k != node_id else outputs
                        for (k, v) in node_outputs.items()
                    }

                    node_ctxs = {
                        k: v if k != node_id else node_ctx
                        for (k, v) in node_ctxs.items()
                    }

                    for (sender_port, receiver_id, receiver_port) in self.graph_edges[
                        node_id
                    ]:
                        receiver_ctx = node_ctxs[receiver_id]
                        copied_node_ctx = TFNodeContext(
                            parameter_ctx=receiver_ctx.parameter_ctx,
                            input_port_ctx={
                                k: v if k != receiver_port else outputs[sender_port]
                                for (k, v) in receiver_ctx.input_port_ctx.items()
                            },
                            metadata=receiver_ctx.metadata,
                        )
                        node_ctxs = {
                            k: v if k != receiver_id else copied_node_ctx
                            for (k, v) in node_ctxs.items()
                        }

        return node_outputs, node_ctxs


class GridSearchNode(TFNode):
    def __init__(self, node: Node, subgraph: TFGraph):
        super(GridSearchNode, self).__init__(node=node)
        self.subgraph = subgraph
        self.var = tf.Variable([[0.0] * 10] * 10, dtype=tf.double)

    @tf.function()
    def __call__(self, node_ctx: TFNodeContext, subgraph: TFGraph):
        for i in range(0, 10):
            subgraph(subgraph.generate_node_contexts())

        return {}, node_ctx


def graph_to_tensorflow(graph: Graph) -> TFGraph:
    return TFGraph(graph=graph)


def model_to_tensorflow(mdf: Model) -> Dict[str, TFGraph]:
    tf_model = {g.get_id(): graph_to_tensorflow(g) for g in mdf.graphs}
    return tf_model


__all__ = ["model_to_tensorflow"]
