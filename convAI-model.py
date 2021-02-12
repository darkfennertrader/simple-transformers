from transformers import (
    OpenAIGPTConfig,
    OpenAIGPTDoubleHeadsModel,
    OpenAIGPTTokenizer,
    TFOpenAIGPTDoubleHeadsModel,
)

config_class, model_class, tokenizer_class = (
    OpenAIGPTConfig,
    OpenAIGPTDoubleHeadsModel,
    OpenAIGPTTokenizer,
)

# configuration = OpenAIGPTConfig(n_positions=1024)
# model = OpenAIGPTDoubleHeadsModel(configuration)
# print(model)

# print()
# print(model.config)


config = OpenAIGPTConfig.from_pretrained(
    "openai-gpt",
    cache_dir="./models/pre-trained/gpt/",
)

model = OpenAIGPTDoubleHeadsModel.from_pretrained(
    "openai-gpt", cache_dir="./models/pre-trained/gpt/"
)
tokenizer = OpenAIGPTTokenizer.from_pretrained(
    "openai-gpt", cache_dir="./models/pre-trained/gpt/"
)

print(model.config)

# freezing the encoder
# for param in model.base_model.parameters():
#     param.requires_grad = False
