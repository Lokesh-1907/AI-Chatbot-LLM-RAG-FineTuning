import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./fine_tuned_model"
OUTPUT_PATH = "./merged_model"

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32
)

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

print("Merging adapter with base model...")

model = model.merge_and_unload()

print("Saving merged model...")

model.save_pretrained(
    OUTPUT_PATH,
    safe_serialization=True
)

tokenizer.save_pretrained(
    OUTPUT_PATH
)

print("Merged model created successfully!")
print("Location:", OUTPUT_PATH)