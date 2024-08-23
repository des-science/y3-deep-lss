# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2024
Author: Arne Thomsen
"""

import tensorflow as tf

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def get_optimizer(net_conf, loss_function="delta_loss", restore_checkpoint=False):
    """
    Get the correctly configured optimizer for the neural network.

    Args:
        net_conf (dict): The configuration dictionary for the neural network, which must be of a specific structure.
        loss_function (str, optional): The loss function to be used, which must be 'delta_loss' or 'likelihood_loss',
            to be used to read the configuration. Defaults to "delta_loss".
        restore_checkpoint (bool, optional): Whether the model has been restored from a checkpoint. Defaults to False.

    Raises:
        NotImplementedError: If the loss function is not implemented.
        NotImplementedError: If the optimizer is not implemented.
        ValueError: If the optimizer is unknown.

    Returns:
        tf.keras.optimizers.Optimizer: The optimizer for the neural network.
    """

    assert not restore_checkpoint, "Handling of models restored from checkpoints is not implemented yet."
    assert loss_function in ["delta", "likelihood", "mutual_info"]
    loss_function = loss_function + "_loss"

    # set up learning rate scheduler
    scheduler = net_conf["optimization"][loss_function]["scheduler"]
    learning_rate = net_conf["optimization"][loss_function]["learning_rate"]
    if scheduler is None:
        learning_rate_schedule = learning_rate
        LOGGER.info(f"Using constant learning rate {learning_rate}")
    elif scheduler == "cosine":
        warmup_init_learning_rate = net_conf["optimization"][loss_function]["warmup_init_learning_rate"]
        warmup_steps = net_conf["optimization"][loss_function]["warmup_steps"]
        decay_steps = net_conf["training"]["n_steps"] - warmup_steps
        end_divided_by_init_learning_rate = net_conf["optimization"][loss_function]["decay_alpha"]

        try:
            learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecay(
                # warmup
                initial_learning_rate=warmup_init_learning_rate,
                warmup_steps=warmup_steps,
                warmup_target=learning_rate,
                # decay
                decay_steps=decay_steps,
                alpha=end_divided_by_init_learning_rate,
            )
        # for TensorFlow 2.9
        except TypeError:
            learning_rate_schedule = LinearWarmupCosineDecaySchedule(
                # warmup
                initial_learning_rate=warmup_init_learning_rate,
                warmup_steps=warmup_steps,
                warmup_target=learning_rate,
                # decay
                decay_steps=decay_steps,
                alpha=end_divided_by_init_learning_rate,
            )
        LOGGER.info(f"Using cosine learning rate schedule with warmup")
    elif scheduler == "warmup":
        warmup_init_learning_rate = net_conf["optimization"][loss_function]["warmup_init_learning_rate"]
        warmup_steps = net_conf["optimization"][loss_function]["warmup_steps"]

        learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=warmup_init_learning_rate,
            decay_steps=warmup_steps,
            end_learning_rate=learning_rate,
            power=1.0,
            cycle=False,
        )
    else:
        raise NotImplementedError(f"Scheduler {scheduler} not implemented yet")

    # set up optimizer
    optimizer_name = net_conf["optimization"]["optimizer"]
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate_schedule, **net_conf["optimization"][loss_function]["optimizer_kwargs"]
        )
        LOGGER.info(f"Using Adam optimizer")
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate_schedule, **net_conf["optimization"][loss_function]["optimizer_kwargs"]
        )
        LOGGER.info(f"Using SGD optimizer")
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        LOGGER.info(f"Rescaling the optimizer for mixed precision")
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    elif tf.keras.mixed_precision.global_policy().name == "mixed_bfloat16":
        raise NotImplementedError("bfloat16 mixed precision not implemented yet")

    return optimizer


class LinearWarmupCosineDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Combined learning rate schedule where first there is a linear warmup, followed by a Cosine decay.

    For TensorFlow 2.15 this is not necessary since the CosineDecay already implements this. But for TensorFlow 2.9 as
    on Perlmutter, we need to implement this ourselves. This custom version should be compatible with the TensorFlow
    2.15 one.
    """

    def __init__(self, initial_learning_rate, warmup_steps, warmup_target, decay_steps, alpha):
        super(LinearWarmupCosineDecaySchedule, self).__init__()

        # warmup
        self.warmup_init_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.learning_rate = warmup_target

        # decay
        self.decay_steps = decay_steps
        self.decay_alpha = alpha

    def __call__(self, step):
        linear_warmup = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.warmup_init_learning_rate,
            decay_steps=self.warmup_steps,
            end_learning_rate=self.learning_rate,
            power=1.0,
            cycle=False,
        )
        cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.learning_rate, decay_steps=self.decay_steps, alpha=self.decay_alpha
        )

        return tf.cond(
            step < self.warmup_steps, lambda: linear_warmup(step), lambda: cosine_decay(step - self.warmup_steps)
        )
