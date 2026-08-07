# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created February 2024
Author: Arne Thomsen
"""

import math

import tensorflow as tf

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def get_optimizer(net_conf, loss_function="delta_loss", restore_checkpoint=False, budget=None):
    """
    Get the correctly configured optimizer for the neural network.

    Args:
        net_conf (dict): The configuration dictionary for the neural network, which must be of a specific structure.
        loss_function (str, optional): The loss function to be used, which must be 'delta_loss' or 'likelihood_loss',
            to be used to read the configuration. Defaults to "delta_loss".
        restore_checkpoint (bool, optional): Whether the model has been restored from a checkpoint. Defaults to False.
        budget (deep_lss.utils.throughput.WallClockBudget, optional): if given with a "cosine" scheduler, the decay is
            driven by elapsed training wall-clock rather than by step index — see
            :class:`WallClockCosineDecay`. The training loop must then call that object once per step. Defaults to
            None, which leaves the step-based behaviour exactly as it was.

    Raises:
        NotImplementedError: If the loss function is not implemented.
        NotImplementedError: If the optimizer is not implemented.
        ValueError: If the optimizer is unknown.

    Returns:
        tf.keras.optimizers.Optimizer: The optimizer for the neural network. When `budget` is used, the returned
        optimizer carries a `wall_clock_schedule` attribute holding the :class:`WallClockCosineDecay` to be stepped.
    """

    # assert not restore_checkpoint, "Handling of models restored from checkpoints is not implemented yet."
    assert loss_function in ["delta", "likelihood", "mutual_info"]
    loss_function = loss_function + "_loss"

    # set up learning rate scheduler
    scheduler = net_conf["optimization"][loss_function]["scheduler"]
    learning_rate = float(net_conf["optimization"][loss_function]["learning_rate"])
    wall_clock_schedule = None
    if scheduler == "cosine" and budget is not None:
        # Wall-clock-driven cosine: warmup stays step-based (so it is identical run to run), the decay
        # spans the remaining time budget. The LR lives in a tf.Variable that the training loop
        # assigns each step, because a keras schedule is a function of optimizer.iterations only.
        wall_clock_schedule = WallClockCosineDecay(
            budget=budget,
            warmup_init_learning_rate=float(net_conf["optimization"][loss_function]["warmup_init_learning_rate"]),
            warmup_steps=net_conf["optimization"][loss_function]["warmup_steps"],
            learning_rate=learning_rate,
            alpha=float(net_conf["optimization"][loss_function]["decay_alpha"]),
        )
        learning_rate_schedule = wall_clock_schedule
        LOGGER.info(
            f"Using a WALL-CLOCK cosine schedule: {wall_clock_schedule.warmup_steps} warmup steps, then cosine "
            f"decay over the remainder of a {budget.total_seconds:.0f} s training budget"
        )
    elif scheduler is None:
        learning_rate_schedule = learning_rate
        LOGGER.info(f"Using constant learning rate {learning_rate}")
    elif scheduler == "cosine":
        warmup_init_learning_rate = float(net_conf["optimization"][loss_function]["warmup_init_learning_rate"])
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
        LOGGER.info("Using cosine learning rate schedule with warmup")
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
    ema_momentum = net_conf["optimization"][loss_function].get("ema_momentum", None)
    if optimizer_name == "adam":
        if ema_momentum is not None:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate_schedule,
                use_ema=True,
                ema_momentum=float(ema_momentum),
                **net_conf["optimization"][loss_function]["optimizer_kwargs"],
            )
            LOGGER.info(f"Using Adam optimizer (non-legacy, EMA momentum={ema_momentum})")
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate_schedule, **net_conf["optimization"][loss_function]["optimizer_kwargs"]
            )
            LOGGER.info("Using Adam optimizer (non-legacy)")
    elif optimizer_name == "adamw":
        # weight_decay is passed through optimizer_kwargs (standard ViT/Swin recipe: AdamW, wd ~0.05)
        if ema_momentum is not None:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate_schedule,
                use_ema=True,
                ema_momentum=float(ema_momentum),
                **net_conf["optimization"][loss_function]["optimizer_kwargs"],
            )
            LOGGER.info(f"Using AdamW optimizer (EMA momentum={ema_momentum})")
        else:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate_schedule,
                **net_conf["optimization"][loss_function]["optimizer_kwargs"],
            )
            LOGGER.info("Using AdamW optimizer")
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate_schedule, **net_conf["optimization"][loss_function]["optimizer_kwargs"]
        )
        LOGGER.info("Using SGD optimizer")
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        LOGGER.info("Wrapping the optimizer in a LossScaleOptimizer for float16 mixed precision")
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    elif tf.keras.mixed_precision.global_policy().name == "mixed_bfloat16":
        # bfloat16 has the same dynamic range as float32, so gradients do not underflow and no
        # loss scaling is needed — use the optimizer unwrapped. The train step skips scaling
        # automatically because it is guarded by isinstance(optimizer, LossScaleOptimizer).
        LOGGER.info("Using bfloat16 mixed precision without loss scaling (not needed)")

    # the training loop needs the handle to step the schedule; hang it off the (possibly wrapped)
    # optimizer so callers do not have to thread a second return value through
    optimizer.wall_clock_schedule = wall_clock_schedule

    return optimizer


class WallClockCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup by step, then cosine decay by elapsed training wall-clock.

    An ordinary ``LearningRateSchedule`` is a pure function of ``optimizer.iterations``, so it cannot
    express "anneal to zero when the allocation runs out" — the step count at which that happens is
    not known when the optimizer is built, and (see :class:`~deep_lss.utils.throughput.WallClockBudget`)
    cannot be reliably predicted from a short probe either. This class instead owns a ``tf.Variable``
    that the training loop assigns once per step from a Python-side computation; ``__call__`` simply
    reads it, so the value the optimizer sees inside its compiled train step is always the current
    one. At a few it/s the assignment is free.

    It has to be a ``LearningRateSchedule`` rather than the bare variable: Keras 3 accepts a schedule
    by reference but *copies* a plain variable into a learning-rate variable of its own, which would
    silently pin the learning rate at its initial value for the whole run.

    Warmup deliberately stays STEP-based: it is a fixed, comparable part of every run, and making it
    depend on the machine's speed would change the optimization trajectory of the early steps for no
    benefit. The cosine then spans the budget that remains once warmup has finished, so the two meet
    without a discontinuity. ``warmup_end_seconds`` is recorded on the budget and persisted, so a
    chained job that restores mid-decay continues the same curve instead of restarting it.

    Args:
        budget (WallClockBudget): the run's training-time budget, already started.
        warmup_init_learning_rate (float): learning rate at step 0.
        warmup_steps (int): number of steps of linear warmup.
        learning_rate (float): peak learning rate, reached at the end of warmup.
        alpha (float): final learning rate as a fraction of the peak, as in ``tf.keras`` CosineDecay.
    """

    def __init__(self, budget, warmup_init_learning_rate, warmup_steps, learning_rate, alpha):
        super().__init__()
        self.budget = budget
        self.warmup_init_learning_rate = float(warmup_init_learning_rate)
        self.warmup_steps = int(warmup_steps)
        self.learning_rate = float(learning_rate)
        self.alpha = float(alpha)
        self.variable = tf.Variable(
            self.warmup_init_learning_rate, trainable=False, dtype=tf.float32, name="wall_clock_lr"
        )

    def __call__(self, step):
        """Read the current learning rate. `step` is ignored: the schedule is driven by `update`."""
        return tf.convert_to_tensor(self.variable)

    def get_config(self):
        return {
            "warmup_init_learning_rate": self.warmup_init_learning_rate,
            "warmup_steps": self.warmup_steps,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
        }

    def value(self, step):
        """Learning rate for `step`, given how much of the time budget has been consumed."""
        if step < self.warmup_steps:
            frac = step / max(self.warmup_steps, 1)
            return self.warmup_init_learning_rate + frac * (self.learning_rate - self.warmup_init_learning_rate)

        # first step past warmup: pin where the decay starts, unless a previous job already did
        if self.budget.warmup_end_seconds is None:
            self.budget.warmup_end_seconds = self.budget.elapsed
            LOGGER.info(
                f"Warmup finished at step {step} after {self.budget.warmup_end_seconds:.0f} s; "
                f"cosine decay now spans the remaining "
                f"{self.budget.total_seconds - self.budget.warmup_end_seconds:.0f} s"
            )

        span = self.budget.total_seconds - self.budget.warmup_end_seconds
        if span <= 0:  # warmup alone already exhausted the budget; nothing left to decay over
            return self.learning_rate * self.alpha
        progress = min(max((self.budget.elapsed - self.budget.warmup_end_seconds) / span, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.learning_rate * (self.alpha + (1.0 - self.alpha) * cosine)

    def update(self, step):
        """Assign the learning rate for `step` and return it."""
        lr = self.value(step)
        self.variable.assign(lr)
        return lr


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
