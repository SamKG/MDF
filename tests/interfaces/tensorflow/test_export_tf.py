import contextlib
import timeit
from modeci_mdf.interfaces.tensorflow.exporter import (
    GridSearchNode,
    model_to_tensorflow,
)

import tensorflow as tf
from modeci_mdf.mdf import Node
from modeci_mdf.utils import load_mdf
from modeci_mdf.execution_engine import EvaluableGraph

# tf.config.run_functions_eagerly(True)


def test_export_abcd():
    abcd_mdf = load_mdf("examples/MDF/ABCD.yaml")
    converted_graphs = model_to_tensorflow(abcd_mdf)
    tf_graph = list(converted_graphs.values())[0]

    tf_results, new_ctx = tf_graph(tf_graph.generate_node_contexts())
    mdf = EvaluableGraph(graph=abcd_mdf.graphs[0])
    # FIXME: this doesn't actually initialize anything?
    mdf.evaluate(initializer={"input0": 0.0})
    mdf_results = {
        k: {p: q.curr_value for (p, q) in v.evaluable_outputs.items()}
        for (k, v) in mdf.enodes.items()
    }
    all_equal = [
        all(
            abs(x - y) <= 0.0000001
            for x, y in zip(tf_results[k].values(), mdf_results[k].values())
        )
        for k in tf_results
    ]

    assert all_equal


def test_gridsearch():
    abcd_mdf = load_mdf("examples/MDF/ABCD.yaml")
    converted_graphs = model_to_tensorflow(abcd_mdf)
    tf_graph = list(converted_graphs.values())[0]

    tf_results, new_ctx = tf_graph(tf_graph.generate_node_contexts())
    gridsearch_node = GridSearchNode(abcd_mdf.graphs[0].nodes[0], tf_graph)
    gridsearch_ctx = gridsearch_node.generate_node_context()

    @contextlib.contextmanager
    def options(options):
        old_opts = tf.config.optimizer.get_experimental_options()
        tf.config.optimizer.set_experimental_options(options)
        try:
            yield
        finally:
            tf.config.optimizer.set_experimental_options(old_opts)

    opts = {
        "layout_optimizer": True,
        "constant_folding": True,
        "shape_optimization": True,
        "remapping": True,
        "arithmetic_optimization": True,
        "dependency_optimization": True,
        "loop_optimization": True,
        "function_optimization": True,
        "debug_stripper": True,
        "disable_model_pruning": True,
        "scoped_allocator_optimization": True,
        "pin_to_host_optimization": True,
        "implementation_selector": True,
        "auto_mixed_precision": True,
    }
    with options(opts):
        outputs, ctx = gridsearch_node(gridsearch_ctx)
        g = tf.autograph.to_graph(gridsearch_node.__call__.python_function)
        print(g)
    # tf_graph.graph_nodes[""]


def test_export_abc_conditions():
    abcd_mdf = load_mdf("examples/MDF/abc_conditions.yaml")
    converted_graphs = model_to_tensorflow(abcd_mdf)
    tf_graph = list(converted_graphs.values())[0]

    tf_results, new_ctx = tf_graph(tf_graph.generate_node_contexts())
    mdf = EvaluableGraph(graph=abcd_mdf.graphs[0])
    # FIXME: this doesn't actually initialize anything?
    mdf.evaluate(initializer={"input0": 0.0})
    mdf_results = {
        k: {p: q.curr_value for (p, q) in v.evaluable_outputs.items()}
        for (k, v) in mdf.enodes.items()
    }
    all_equal = [
        all(
            abs(x - y) <= 0.0000001
            for x, y in zip(tf_results[k].values(), mdf_results[k].values())
        )
        for k in tf_results
    ]

    assert all_equal


def test_export_abcd_lca():
    abcd_mdf = load_mdf("examples/MDF/ABCD_LCA.yaml")
    converted_graphs = model_to_tensorflow(abcd_mdf)
    tf_graph = list(converted_graphs.values())[0]

    @contextlib.contextmanager
    def options(options):
        old_opts = tf.config.optimizer.get_experimental_options()
        tf.config.optimizer.set_experimental_options(options)
        try:
            yield
        finally:
            tf.config.optimizer.set_experimental_options(old_opts)

    opts = {
        "layout_optimizer": True,
        "constant_folding": True,
        "shape_optimization": True,
        "remapping": True,
        "arithmetic_optimization": True,
        "dependency_optimization": True,
        "loop_optimization": True,
        "function_optimization": True,
        "debug_stripper": True,
        "disable_model_pruning": True,
        "scoped_allocator_optimization": True,
        "pin_to_host_optimization": True,
        "implementation_selector": True,
        "auto_mixed_precision": True,
    }
    with options(opts):
        with tf.device("/CPU:0"):
            start_t = timeit.default_timer()
            call_fun = tf.autograph.to_code(tf_graph.__call__.python_function)
            print(call_fun)
            node_contexts = tf_graph.generate_node_contexts()
            result = tf_graph(node_contexts)
            end_t = timeit.default_timer()
            warmup_t = end_t - start_t
            total_t = 0
            for i in range(0, 5):
                start_t = timeit.default_timer()
                result = tf_graph(node_contexts)
                end_t = timeit.default_timer()
                total_t += end_t - start_t
            total_t /= 5

    print(result, total_t)


if __name__ == "__main__":
    test_export_abcd_lca()
