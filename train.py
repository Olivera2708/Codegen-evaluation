import pandas as pd
import matplotlib.pyplot as plt
from peft import LoraConfig
from datasets import Dataset
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


class LossLoggingCallback(TrainerCallback):
    def __init__(self):
        self.loss_values = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            self.loss_values.append(logs['loss'])
            # print(f"Logged Loss (on_log): {logs['loss']}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history and 'loss' in state.log_history[-1]:
            self.loss_values.append(state.log_history[-1]['loss'])
            # print(f"Logged Loss (on_step_end): {state.log_history[-1]['loss']}")


def train_model(device, model, tokenizer):
    model.train()
    tokenizer.padding_side = 'right'

    data_train = pd.read_json('data_train.json')
    data_eval = pd.read_json('data_eval.json')
    dataset_train = Dataset.from_pandas(data_train)
    dataset_eval = Dataset.from_pandas(data_eval)

    training_args = TrainingArguments(
        output_dir='./model',
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=5,
        logging_steps=1,
        fp16=True,
        report_to=None
    )

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example)):
            text = f"<|system|>Write a function without explanation.\n<|user|>You are an expert Kotlin programmer, and here is your task. {example['prompt'][i]}\n<|assistant|>\n{example['solution'][i]}<|endoftext|>"
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
        peft_config=qlora_config,
        max_seq_length=256,
        data_collator=collator,
        formatting_func=formatting_prompts_func
    )

    loss_logging_callback = LossLoggingCallback()
    trainer.add_callback(loss_logging_callback)

    trainer.train()

    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')


    if not loss_logging_callback.loss_values:
        print("No loss values logged. Check if dataset size is large enough.")
    else:
        plt.plot(range(len(loss_logging_callback.loss_values)), loss_logging_callback.loss_values, label='Training Loss')
        plt.xlabel('Logging Step')
        plt.ylabel('Loss')
        plt.title('Training Loss over Time')
        plt.legend()
        plt.show()
