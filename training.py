import os
import time
import torch
import pickle
import numpy as np
import pandas as pd
import GPUtil
import json

# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     AutoModelForCausalLM,
# )
# import optuna

# from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from simpletransformers.conv_ai import ConvAIModel, ConvAIArgs

cuda_available = torch.cuda.is_available()

#######################   GPU AVAILABILITY   ################################
device = torch.device("cuda" if cuda_available else "cpu")
if cuda_available:
    print("\nGPU is OK")
else:
    print("\nGPU is KO")
print()
print(GPUtil.showUtilization())
#############################################################################

model_args = ConvAIArgs()
model_args.wandb_project = "logs_training"
model_args.num_train_epochs = 1
model_args.overwrite_output_dir = True
model_args.fp16 = True
model_args.output_dir = "./models/fine-tuned/gpt/"
model_args.train_batch_size = 4
model_args.max_seq_length = 128
model_args.num_candidates = 19

# print(dir(model_args))


test_file = "./models/training_data/test_data.json"


# Initialization for GPT2
model = ConvAIModel(
    model_type="gpt2",  # # values: ["gpt", "gpt2"]
    model_name="gpt2",  # values: ["openai-gpt", "gpt2"]
    cache_dir="./models/pre-trained/gpt2/gpt2-small",  # locally pre-loaded model
    use_cuda=cuda_available,
    args=model_args,
)

# Initialization for GPT
# model = ConvAIModel(
#     model_type="gpt",  # # values: ["gpt", "gpt2"]
#     model_name="openai-gpt",  # values: ["openai-gpt", "gpt2", <local_dir>]
#     cache_dir="./models/pre-trained/gpt/",  # locally pre-loaded model
#     use_cuda=cuda_available,
#     args=model_args,
# )

with open(test_file, "r", encoding="utf-8") as f:
    raw_data = json.loads(f.read())

dataset = dict()
dataset = [raw_data]

# print(dataset)

with open("./models/training_data/dataset.json", "w") as f:
    json.dump(dataset, f)


# with open("dataset.json", "r", encoding="utf-8") as f:
#     raw_data = json.loads(f.read())


model.train_model(
    train_file="./models/training_data/dataset.json",
    show_running_loss=True,
    eval_file=False,
)

##########################################################
# def greet_me(**kwargs):
#     print(type(kwargs))
#     for key, value in kwargs.items():
#         print("{0} = {1}".format(key, value))


# greet_me(name="yasoob")
##########################################################