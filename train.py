import pandas as pd
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments


def train_model(device, model, tokenizer):
    data_train = pd.read_json('data_train.json')
    data_eval = pd.read_json('data_eval.json')
    dataset_train = Dataset.from_pandas(data_train)
    dataset_eval = Dataset.from_pandas(data_eval)

    def tokenize_function(examples):
        model_inputs = tokenizer(examples['prompt'], max_length=128, truncation=True, padding = "max_length", return_tensors='pt')
        labels = tokenizer(examples['solution'], max_length=128, truncation=True, padding = "max_length", return_tensors='pt')

        model_inputs['labels'] = labels['input_ids']
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        return model_inputs

    dataset_train = dataset_train.map(tokenize_function, batched=True)
    dataset_eval = dataset_eval.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        fp16=True,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval
    )

    torch.cuda.empty_cache()
    trainer.train()

    model.save_pretrained('./fine-tuned-model')
    tokenizer.save_pretrained('./fine-tuned-model')

# # Function to generate code solutions from prompts
# def generate_solution(prompt):
#     input_ids = tokenizer.encode(prompt, return_tensors='pt')
#     output = model.generate(input_ids, max_length=100, num_return_sequences=1)
#     return tokenizer.decode(output[0], skip_special_tokens=True)

# # Example usage of the generation function
# if __name__ == "__main__":
#     new_prompt = "Write a function in Python to check if a number is prime."
#     generated_solution = generate_solution(new_prompt)
#     print("Generated Solution:\n", generated_solution)
