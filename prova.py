from args import Args
import wandb

from transformers import AutoModelForCausalLM


def sweep_config_to_sweep_values(sweep_config):
    """
    Converts an instance of wandb.Config to plain values map.
    wandb.Config varies across versions quite significantly,
    so we use the `keys` method that works consistently.
    """

    return {key: sweep_config[key] for key in sweep_config.keys()}


args = Args(
    do_train=True,
    do_eval=False,
    num_train_epochs=1,
    save_steps=12000,
    wandb_project="steve-chatbot",
)


def wandb_conf(args):
    config = {
        "model_type": args.model_type,
        "model_name_or_path": args.model_name_or_path,
        "train_epochs": args.per_gpu_train_batch_size,
        "learning_rate": 0.001,
    }
    return config


print(args.wandb_project)
# model_class = AutoModelForCausalLM
# model_name = args.model_name_or_path
# wandb_model = model_class.from_pretrained(model_name)
# if args.wandb_project:
#     wandb.init(project=args.wandb_project, config=wandb_conf(args))
#     wandb.watch(wandb_model)

# wandb.init(project=args.wandb_project).finish()

import math
import random

# Start a new run, tracking config metadata
wandb.init(
    project=args.wandb_project,
    config=wandb_conf(args),
)
config = wandb.config

# Simulating a training or evaluation loop
for x in range(1000):
    acc = math.log(1 + x + random.random() * config.learning_rate) + random.random()
    loss = (
        10
        - math.log(1 + x + random.random() + config.learning_rate * x)
        + random.random()
    )
    # Log metrics from your script to W&B
    wandb.log({"acc": acc, "loss": loss})

wandb.finish()
