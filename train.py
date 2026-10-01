from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "training_data.jsonl"
OUTPUT_DIR = "./fine_tuned_model"

print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train"
)

def format_example(example):
    return (
        "### Instruction:\n"
        + example["instruction"]
        + "\n\n"
        "### Input:\n"
        + example["input"]
        + "\n\n"
        "### Response:\n"
        + example["output"]
    )

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none"
)

print("Starting fine-tuning...")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,
    formatting_func=format_example
)

trainer.train()

print("Saving fine-tuned model...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning completed successfully!")