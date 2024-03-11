# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created January 2024
Author: Arne Thomsen
"""

import tensorflow as tf


def write_summary(label, value, summary_writer, training=True, summary_type="scalar", print_scalar=False):
    """Handle different kinds of summaries to TensorBoard.

    Args:
        label (str): The name of the summary.
        value (tf.Tensor): The value to log.
        summary_writer (tf.summary.SummaryWriter): The summary writer.
        training (bool, optional): Only log during training. Defaults to True.
        summary_type (str, optional): The kind of summary, allowed are 'scalar', 'histogram' and 'image. Defaults to
            "scalar".
        print_scalar (bool, optional): Print the scalar value to the console. Defaults to False.

    Raises:
        ValueError: If an invalid summary_type is passed.
    """
    if summary_writer is not None and training:
        with summary_writer.as_default():
            if summary_type == "scalar":
                tf.summary.scalar(label, value)
                if print_scalar:
                    tf.print(f"{label}: {value}")
            elif summary_type == "histogram":
                tf.summary.histogram(label, value)
            elif summary_type == "image":
                # value = tf.image.resize(value, size=(128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                tf.summary.image(label, value)
            else:
                raise ValueError(f"Invalid summary type {summary_type} was passed")
