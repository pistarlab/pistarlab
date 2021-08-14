import math

from ray.rllib.agents.a3c.a3c import DEFAULT_CONFIG as A3C_CONFIG, \
    validate_config, get_policy_class
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.execution.train_ops import ComputeGradients, AverageGradients, \
    ApplyGradients, TrainOneStep
from ray.rllib.utils import merge_dicts

from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG
from .ac_policy import PistarACTorchPolicy
from .default_vision_model import VisionModel

Pistar_A2CDEFAULT_CONFIG = merge_dicts(A2C_DEFAULT_CONFIG,{
    'model':{
        'custom_model': "pistarvisionnet",
        'custom_model_config':{}
    }

})

def execution_plan(workers, config):
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    if config["microbatch_size"]:
        num_microbatches = math.ceil(
            config["train_batch_size"] / config["microbatch_size"])
        # In microbatch mode, we want to compute gradients on experience
        # microbatches, average a number of these microbatches, and then apply
        # the averaged gradient in one SGD step. This conserves GPU memory,
        # allowing for extremely large experience batches to be used.
        train_op = (
            rollouts.combine(
                ConcatBatches(min_batch_size=config["microbatch_size"]))
            .for_each(ComputeGradients(workers))  # (grads, info)
            .batch(num_microbatches)  # List[(grads, info)]
            .for_each(AverageGradients())  # (avg_grads, info)
            .for_each(ApplyGradients(workers)))
    else:
        # In normal mode, we execute one SGD step per each train batch.
        train_op = rollouts \
            .combine(ConcatBatches(
                min_batch_size=config["train_batch_size"])) \
            .for_each(TrainOneStep(workers))

    return StandardMetricsReporting(train_op, workers, config)


PistarA2CTrainer = build_trainer(
    name="PistarA2C",
    default_config=Pistar_A2CDEFAULT_CONFIG,
    default_policy=PistarACTorchPolicy,
    get_policy_class=lambda x: PistarACTorchPolicy,
    validate_config=validate_config,
    execution_plan=execution_plan)
