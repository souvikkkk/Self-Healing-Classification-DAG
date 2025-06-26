from datasets import load_dataset
from transformers import (
AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
import torch

# 1. Load dataset
dataset = load_dataset("imdb")
train_data = dataset["train"].shuffle(seed=42).select(range(2000))  # Use subset for quick training
test_data = dataset["test"].shuffle(seed=42).select(range(500))

# 2. Tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

train_data = train_data.map(tokenize, batched=True)
test_data = test_data.map(tokenize, batched=True)
train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 3. Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4. Apply LoRA
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(model, config)

import transformers
print(transformers.__file__)
print(transformers.TrainingArguments.__module__)


# 5. Training args
training_args = TrainingArguments(
    output_dir="./models",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data
)

# 7. Train
trainer.train()

# 8. Save the model
model.save_pretrained("models/lora-distilbert-imdb")
tokenizer.save_pretrained("models/lora-distilbert-imdb")
