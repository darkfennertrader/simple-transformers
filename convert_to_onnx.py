from pathlib import Path
from transformers.convert_graph_to_onnx import convert
from simpletransformers.conv_ai import ConvAIModel, ConvAIArgs
from transformers import (
    WEIGHTS_NAME,
    GPT2Config,
    GPT2DoubleHeadsModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTDoubleHeadsModel,
    OpenAIGPTTokenizer,
)

# from onnxruntime_tools.transformers import convert_to_onnx

# """
# This converts GPT2 model to onnx. Examples:
# (1) Convert pretrained model 'gpt2' to ONNX
#    python convert_to_onnx.py -m gpt2 --output gpt2.onnx
# (2) Convert pretrained model 'distilgpt2' to ONNX, and use optimizer to get float16 model.
#    python convert_to_onnx.py -m distilgpt2 --output distilgpt2_fp16.onnx -o -p fp16
# (3) Convert a model check point to ONNX, and run optimization and int8 quantization
#    python convert_to_onnx.py -m ./my_model_checkpoint/ --output my_model_int8.onnx -o -p int8
# """


# MODEL_CLASSES = {
#     "gpt": (OpenAIGPTConfig, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer),
#     "gpt2": (GPT2Config, GPT2DoubleHeadsModel, GPT2Tokenizer),
# }


# loading chatbot fine-tuned model in the GPU if possibile
model_args = ConvAIArgs()
model_args.manual_seed = 42  # set seed for reproducibility
model_args.max_history = 1
model_args.max_length = 300
model_args.do_sample = True
model_args.temperature = 0.7
model_args.top_k = 100
model_args.top_p = 0.9

model_type = "gpt"
model_name = "./models/fine-tuned/gpt/"

model = ConvAIModel(
    model_type=model_type,
    model_name=model_name,
    use_cuda=False,
    args=model_args,
)

# config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

# model = model_class.from_pretrained("./models/fine-tuned/gpt/")
# print(type(model))

# print(config_class())

# convert(
#     framework="pt",
#     model=model,
#     output=Path("./models/onnx/gpt_tuned.onnx"),
#     opset=13,
#     # use_external_format=True,
# )