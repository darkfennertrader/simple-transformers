import os
from pathlib import Path
from transformers.convert_graph_to_onnx import convert

# from simpletransformers.conv_ai import ConvAIModel, ConvAIArgs
from transformers import (
    WEIGHTS_NAME,
    GPT2Config,
    GPT2DoubleHeadsModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTDoubleHeadsModel,
    OpenAIGPTTokenizer,
)
import onnx
import onnxruntime
from onnxruntime_tools import optimizer
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime_tools.transformers.gpt2_helper import (
    Gpt2Helper,
    MyGPT2LMHeadModel,
    MyGPT2LMHeadModel_NoPadding,
)

################################################################################
# (1) Exporting GPT2 fine-tuned model to ONNX fp32

# device = "cpu"
# model = MyGPT2LMHeadModel.from_pretrained("./models/fine-tuned/dialogpt/small/")
# Gpt2Helper.export_onnx(
#     model,
#     device,
#     "./models/onnx/dialogpt/small/dialogpt_ft_small.onnx",
#     use_external_data_format=True,
#     verbose=True,
# )


################################################################################
# (2) Model OPTIMIZATION (optional fp16)

# onnx_model_path = "./models/onnx/dialogpt/small/dialogpt_ft_small.onnx"
# optimized_model_path = "./models/onnx/dialogpt/small/dialogpt_ft_opt_small.onnx"
# model = MyGPT2LMHeadModel.from_pretrained("./models/fine-tuned/dialogpt/small/")
# config = model.config
# is_float16 = True  # conversion to mixed precision
# use_gpu = True

# Gpt2Helper.optimize_onnx(
#     onnx_model_path,
#     optimized_model_path,
#     is_float16,
#     config.num_attention_heads,
#     config.n_layer,
# )

# print(
#     "ONNX full precision model size (MB):",
#     os.path.getsize("./models/onnx/dialogpt/small/dialogpt_ft_small.onnx")
#     / (1024 * 1024),
# )
# print(
#     "ONNX optimized model size (MB):",
#     os.path.getsize("./models/onnx/dialogpt/small/dialogpt_ft_opt_small.onnx")
#     / (1024 * 1024),
# )


################################################################################
# (3) Model QUANTIZATION
# def quantize_onnx_model(onnx_model_path, quantized_model_path):
#     onnx_opt_model = onnx.load(onnx_model_path)
#     quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QInt8)


# quantize_onnx_model(
#     "./models/onnx/dialogpt/small/dialogpt_ft_small.onnx",
#     "./models/onnx/dialogpt/small/dialogpt_ft_quant_small.onnx",
# )

# print(
#     "ONNX full precision model size (MB):",
#     os.path.getsize("./models/onnx/dialogpt/small/dialogpt_ft_small.onnx")
#     / (1024 * 1024),
# )
# print(
#     "ONNX quantized model size (MB):",
#     os.path.getsize("./models/onnx/dialogpt/small/dialogpt_ft_quant_small.onnx")
#     / (1024 * 1024),
# )

################################################################################
# (4) MODEL PERFORMANCE

# model = MyGPT2LMHeadModel.from_pretrained("./models/fine-tuned/dialogpt/small/")
# config = model.config

# dummy_inputs = Gpt2Helper.get_dummy_inputs(
#     batch_size=1,
#     past_sequence_length=1,
#     sequence_length=1,
#     num_attention_heads=config.num_attention_heads,
#     hidden_size=config.hidden_size,
#     num_layer=config.n_layer,
#     vocab_size=config.vocab_size,
#     device="cpu",
#     float16=False,
#     has_position_ids=True,
#     has_attention_mask=True,
# )
# # Inference with Pytorch
# output, _ = Gpt2Helper.pytorch_inference(model, dummy_inputs, total_runs=10)
# # Inference with ONNX
# ort_session = onnxruntime.InferenceSession(
#     "./models/onnx/dialogpt/small/dialogpt_ft_small.onnx"
# )
# output, _ = Gpt2Helper.onnxruntime_inference(ort_session, dummy_inputs, total_runs=10)

################################################################################
# (5) MODEL INFERENCE

model = MyGPT2LMHeadModel_NoPadding.from_pretrained(
    "./models/fine-tuned/dialogpt/small/"
)
config = model.config

ort_session = onnxruntime.InferenceSession(
    "./models/onnx/dialogpt/small/dialogpt_ft_small.onnx"
)