"""
Functions for converting from MDF models to PyTorch models.
"""
from platform import node
from typing import Callable, Dict, List, Tuple, Union
import tensorflow as tf
from modeci_mdf.execution_engine import EvaluableGraph

from modeci_mdf.interfaces.tensorflow.builtins import get_builtin
from modeci_mdf.mdf import Condition, ConditionSet, Edge, Graph, Model, Node, Parameter


def parameter_to_tensorflow(parameter: Parameter) -> tf.Operation:
    if (function := getattr(parameter, "function")) is not None:
        return get_builtin(function)
    elif (value := getattr(parameter, "value")) is not None:
        return tf.Variable(value, name=parameter.get_id(), dtype=tf.double)
    else:
        raise Exception(f"Don't know how to convert parameter {parameter}!")


class TFNode(tf.Module):
    def __init__(self, node: Node):
        super(TFNode, self).__init__(name=node.get_id())

        self.node_parameters = {
            p.get_id(): parameter_to_tensorflow(p) for p in node.parameters
        }

        self.input_ports = {
            p.get_id(): tf.Variable(
                0.0, dtype=tf.double, shape=tf.TensorShape(None), name=p.get_id()
            )
            for p in node.input_ports
        }

        self.node_keyvalues = dict(**self.node_parameters, **self.input_ports)

        func_args = {}
        for f in [f for f in node.parameters if getattr(f, "function") is not None]:
            f_args = {}
            for (arg, val) in f.args.items():
                f_args[arg] = self.node_keyvalues[val]
            func_args[f.get_id()] = f_args

        self.metadata = {
            "execution_count": tf.Variable(0, dtype=tf.int32, name="execution_count")
        }

        self.nonfunc_node_parameters = [
            p for (p, v) in self.node_parameters.items() if isinstance(v, tf.Variable)
        ]
        self.func_args = func_args
        self.output_port_mappings = {o.get_id(): o.value for o in node.output_ports}

    @tf.function
    def __call__(self):
        self.metadata["execution_count"].assign_add(1)

        func_outputs = {}
        for (func_id, args) in self.func_args.items():
            func_outputs[func_id] = self.node_parameters[func_id](**args)
        for parameter in self.nonfunc_node_parameters:
            func_outputs[parameter] = self.node_parameters[parameter]

        output_ports = {}
        for (o, v) in self.output_port_mappings.items():
            output_ports[o] = func_outputs[v]

        return output_ports


def node_to_tensorflow(node: Node) -> Union[Callable, tf.Variable]:
    return TFNode(node)


class TFGraph(tf.Module):
    def __init__(self, graph: Graph):
        super(TFGraph, self).__init__(name=graph.get_id())
        scheduler = EvaluableGraph(graph=graph).scheduler
        self.consideration_queue = [
            set(n.get_id() for n in s) for s in scheduler.consideration_queue
        ]

        self.node_conditions = {n.get_id(): {} for n in graph.nodes}

        self.graph_nodes: Dict[str, TFNode] = {}
        for n in graph.nodes:
            self.graph_nodes[n.get_id()] = node_to_tensorflow(n)
        self.graph_edges: Dict[str, Tuple[str, str, str]] = {
            n.get_id(): [] for n in graph.nodes
        }
        for edge in graph.edges:
            self.graph_edges[edge.sender].append(
                (edge.sender_port, edge.receiver, edge.receiver_port)
            )

    @tf.function
    def __call__(self, inputs: Dict[str, Dict[str, tf.Tensor]]):
        # Set initial values for run
        for (id, value) in inputs.items():
            for (key, tensor) in value.items():
                self.graph_nodes[id].node_keyvalues[key].assign(tensor)

        active_nodes = {}
        for node_id in self.graph_nodes.keys():
            active_nodes[node_id] = True

        node_outputs = {}
        for exec_set in self.consideration_queue:
            for node_id in exec_set:
                if not active_nodes[node_id]:
                    continue
                outputs = self.graph_nodes[node_id]()
                node_outputs[node_id] = outputs
                for (sender_port, receiver_id, receiver_port) in self.graph_edges[
                    node_id
                ]:
                    self.graph_nodes[receiver_id].input_ports[receiver_port].assign(
                        outputs[sender_port]
                    )
                conditions = self.node_conditions[node_id]

        return node_outputs


def graph_to_tensorflow(graph: Graph) -> Callable:
    return TFGraph(graph=graph)


def model_to_tensorflow(mdf: Model) -> Dict[str, tf.Graph]:
    tf_model = {g.get_id(): graph_to_tensorflow(g) for g in mdf.graphs}
    return tf_model


__all__ = ["model_to_tensorflow"]
