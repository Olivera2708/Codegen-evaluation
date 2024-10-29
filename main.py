from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch
from evaluation import evaluate_model_on_kotlin_humaneval
from train import train_model


dataset = load_dataset("JetBrains/Kotlin_HumanEval")["train"]
device = "cuda" if torch.cuda.is_available() else "cpu"

def train_fine_tuned_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3b-code-base-2k")
    model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3b-code-base-2k", quantization_config=bnb_config, trust_remote_code=True).to(device)

    train_model(device, model, tokenizer)

def evaluate_original():
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3b-code-base-2k")
    model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3b-code-base-2k").to(device)
    model.eval()
    result, executable_code, broken_code = evaluate_model_on_kotlin_humaneval(device, model, tokenizer, dataset)

    print(f"Executable code -> {executable_code}")
    print(f"Broken code -> {broken_code}")
    print(f"Good results from executable -> {result*100}%")

def evaluate_fine_tuned():
    tokenizer = AutoTokenizer.from_pretrained("./model")
    model = AutoModelForCausalLM.from_pretrained("./model").to(device)
    model.eval()

    result, executable_code, broken_code = evaluate_model_on_kotlin_humaneval(device, model, tokenizer, dataset)
    print(f"Executable code -> {executable_code}")
    print(f"Broken code -> {broken_code}")
    print(f"Good results from executable -> {result*100}%")


if __name__ == "__main__":
    # print("--- ORIGINAL MODEL ---")
    # evaluate_original()

    # train_fine_tuned_model()

    print("--- FINE-TUNED MODEL ---")
    evaluate_fine_tuned()