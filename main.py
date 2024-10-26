from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from evaluation import evaluate_model_on_kotlin_humaneval


device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3b-code-base-2k")
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3b-code-base-2k").to(device)
model.eval()

dataset = load_dataset("JetBrains/Kotlin_HumanEval")["train"]
result, executable_code, broken_code = evaluate_model_on_kotlin_humaneval(device, model, tokenizer, dataset, k_values=[1, 5, 10])

print(f"Executable code -> {executable_code}")
print(f"Broken code -> {broken_code}")
print(f"Godo results from executable -> {result*100}%")