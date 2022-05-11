import contextlib
import timeit
import numpy as np
from modeci_mdf.interfaces.tensorflow.exporter import model_to_tensorflow

import tensorflow as tf
from modeci_mdf.utils import load_mdf
from modeci_mdf.execution_engine import EvaluableGraph

# tf.config.run_functions_eagerly(True)

def test_export_abcd():
    abcd_mdf = load_mdf("examples/MDF/ABCD.yaml")
    converted_graphs = model_to_tensorflow(abcd_mdf)
    tf_graph = list(converted_graphs.values())[0]

    tf_results = tf_graph(
        {"input0": {"input_level": tf.constant(0.0, dtype=tf.double)}}
    )
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

def test_export_abc_conditions():
    abcd_mdf = load_mdf("examples/MDF/abc_conditions.yaml")
    converted_graphs = model_to_tensorflow(abcd_mdf)
    tf_graph = list(converted_graphs.values())[0]

    tf_results = tf_graph(
        {"input0": {"input_level": tf.constant(0.0, dtype=tf.double)}}
    )
    mdf = EvaluableGraph(graph=abcd_mdf.graphs[0])
    # FIXME: this doesn't actually initialize anything?
    mdf.evaluate(initializer={"input0": 0.0})
    mdf_results = {
        k: {p: q.curr_value for (p, q) in v.evaluable_outputs.items()}
        for (k, v) in mdf.enodes.items()
    }
    # FIXME: significant difference in results (precision related?)
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
            result = tf_graph({"A": tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.double)})
            end_t = timeit.default_timer()
            warmup_t = end_t - start_t
            total_t = 0
            for i in range(0, 5):
                start_t = timeit.default_timer()
                result = tf_graph(
                    {"A": tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.double)}
                )
                end_t = timeit.default_timer()
                total_t += end_t - start_t
            total_t /= 5

    print(result, total_t)


if __name__ == "__main__":
    test_export_abcd_lca()
