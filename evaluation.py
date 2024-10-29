from sacrebleu import corpus_bleu
import torch
from test import run_tests

def exact_match_score(prediction, reference):
    return int(prediction.strip() == reference.strip())

def bleu_score(predictions, references):
    return corpus_bleu(predictions, [references]).score

def evaluate_model_on_kotlin_humaneval(device, model, tokenizer, dataset, max_length=512):
    sum_good = 0
    sum_all = 0
    broken_code = 0
    executable_code = 0
    for sample in dataset:
        prompt = sample['prompt']
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(inputs, max_length=max_length)
        
        generated_code = tokenizer.decode(output[0], skip_special_tokens=True)

        # print(generated_code)

        value, good, all = run_tests(generated_code, sample['test'])
        if value:
            sum_good += good
            sum_all += all
            executable_code += 1
        else:
            broken_code += 1

        if sum_all == 0:
            sum_all = 1
    return sum_good/sum_all, executable_code, broken_code
