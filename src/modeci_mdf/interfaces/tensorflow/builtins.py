import sys
from pyrsistent import b
import tensorflow as tf


@tf.function(experimental_follow_type_hints=True, jit_compile=True)
def linear(variable0: tf.Tensor, slope: tf.Tensor, intercept: tf.Tensor):
    return (variable0 * slope) + intercept


@tf.function(experimental_follow_type_hints=True, jit_compile=True)
def logistic(variable0: tf.Tensor, gain: tf.Tensor, bias: tf.Tensor, offset: tf.Tensor):
    # we use negative offset since sigmoid negates entire exponent
    return tf.sigmoid(gain * (variable0 + bias) - offset)


@tf.function(experimental_follow_type_hints=True, jit_compile=True)
def exponential(
    variable0: tf.Tensor,
    scale: tf.Tensor,
    rate: tf.Tensor,
    bias: tf.Tensor,
    offset: tf.Tensor,
):
    return scale * tf.exp(rate * variable0 + bias) + offset


@tf.function(experimental_follow_type_hints=True, jit_compile=True)
def sin(
    variable0: tf.Tensor,
    scale: tf.Tensor,
) -> tf.Tensor:
    return scale * tf.sin(variable0)


@tf.function(experimental_follow_type_hints=True, jit_compile=True)
def lca(
    variable0: tf.Tensor,
    leak: tf.Tensor,
    competition: tf.Tensor,
    threshold: tf.Tensor,
    time_step: tf.Tensor,
    max_iterations: tf.Tensor,
) -> tf.Tensor:
    max_iterations = tf.cast(max_iterations, tf.int32)
    active = tf.ones_like(variable0)
    rts = tf.zeros_like(variable0)

    variable_shape = tf.expand_dims(
        tf.reduce_sum(tf.ones_like(variable0, dtype=tf.int32)), 0
    )

    loop_ct = tf.constant(0, dtype=tf.int32)

    def lca_body(
        loop_ct: tf.Tensor,
        variable0: tf.Tensor,
        active: tf.Tensor,
        rts: tf.Tensor,
        leak: tf.Tensor,
        competition: tf.Tensor,
        threshold: tf.Tensor,
        time_step: tf.Tensor,
    ):
        dp_competition = tf.reduce_sum(variable0) - variable0
        dp = -leak * variable0 - competition * dp_competition
        rts = rts + active
        active = tf.where(
            tf.abs(variable0) < threshold, tf.ones_like(rts), tf.zeros_like(rts)
        )
        dw = tf.random.normal(variable_shape, dtype=tf.double)
        left = dp * time_step
        right = dw * tf.sqrt(time_step)
        variable0 = variable0 + active * (left + right)
        # tf.print("step!", active, variable0, sys.stdout)
        loop_ct = tf.add(loop_ct, 1)
        return (
            loop_ct,
            variable0,
            active,
            rts,
            leak,
            competition,
            threshold,
            time_step,
        )

    (
        loop_ct,
        variable0,
        active,
        rts,
        leak,
        competition,
        threshold,
        time_step,
    ) = tf.while_loop(
        lambda ctr, x0, x1, x2, x3, x4, x5, x6: True,
        lca_body,
        (loop_ct, variable0, active, rts, leak, competition, threshold, time_step),
        parallel_iterations=1,
        shape_invariants=None,
        maximum_iterations=max_iterations,
    )
    # tf.print(loop_ct, max_iterations, output_stream=sys.stdout)

    rts = rts * time_step
    return rts


builtin_map = {
    "linear": linear,
    "logistic": logistic,
    "exponential": exponential,
    "sin": sin,
    "lca": lca,
}


def get_builtin(name: str):
    if name not in builtin_map:
        raise Exception(
            f"Tensorflow interface currently doesn't support builtin operation {name}!"
        )
    return builtin_map[name]
