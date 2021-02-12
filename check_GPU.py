import os

import transformers
import simpletransformers

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# import tensorflow as tf
import torch

# print()
# print(tf.__version__)
# older versions of tensorflow
# tf.test.is_gpu_available( cuda_only=False, min_cuda_compute_capability=None)
# print(f"Is Tensorflow using GPU?: { tf.config.list_physical_devices('GPU') }")
print()
print(torch.__version__)
print(f"Is Pytorch using GPU?:  {torch.cuda.is_available()}")
print()