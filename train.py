import pandas as pd
import torch
from peft import LoraConfig
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def train_model(device, model, tokenizer):
    model.train()
    data_train = pd.read_json('data_train.json')
    data_eval = pd.read_json('data_eval.json')
    dataset_train = Dataset.from_pandas(data_train)
    dataset_eval = Dataset.from_pandas(data_eval)

    training_args = TrainingArguments(
        output_dir='./model',
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        logging_steps=100,
        fp16=True
    )

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example)):
            text = f"<|system|>\nYou are an expert Kotlin programmer, and here is your task.\n<|user|>\n{example['prompt'][i]}\n<|assistant|>\n{example['solution'][i]}<|endoftext|>"
            output_texts.append(text)
        return output_texts

    qlora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none"
    )

    response_template = "\n<|assistant|>\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        tokenizer=tokenizer,
        peft_config = qlora_config,
        max_seq_length=256,
        data_collator=collator,
        formatting_func=formatting_prompts_func
    )

    torch.cuda.empty_cache()
    trainer.train()

    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')
