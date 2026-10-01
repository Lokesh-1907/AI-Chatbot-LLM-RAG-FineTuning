import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./fine_tuned_model"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32
)

print("Loading fine-tuned adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

model.eval()

question = input("Enter your question: ")

prompt = f"""
### Instruction:
{question}

### Input:

### Response:
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nFine-tuned model response:")
print(answer)