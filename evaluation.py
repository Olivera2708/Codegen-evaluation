from test import run_tests

def clean_prompt(prompt: str) -> str:
    cleaned_lines = [
        line.strip().replace('*', '').replace("/", "").strip()
        for line in prompt.split('\n')
    ]

    cleaned_prompt = ' '.join(cleaned_lines)
    return cleaned_prompt

def extract_first_kotlin_code(response):
    start_indicator = "```kotlin"
    end_indicator = "```"

    in_code_block = False
    kotlin_code_lines = []

    for line in response.splitlines():
        if line.strip() == start_indicator:
            if not in_code_block:
                in_code_block = True
                continue

        if in_code_block:
            if line.strip() == end_indicator:
                return "\n".join(kotlin_code_lines)
            kotlin_code_lines.append(line)
    return ""

def evaluate_model_on_kotlin_humaneval(device, model, tokenizer, dataset, max_length=512):
    sum_good = 0
    sum_all = 0
    broken_code = 0
    executable_code = 0
    for sample in dataset:
        prompt = f"<|system|>Write a function without explanation.\n<|user|>{clean_prompt(sample['prompt'])}\n<|assistant|>\n"
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(inputs, max_length=max_length)
        
        generated_code = tokenizer.decode(output[0], skip_special_tokens=True)

        print(output[0])
        print(generated_code)
        code = extract_first_kotlin_code(generated_code)

        value, good, all = run_tests(code, sample['test'])
        if value:
            sum_good += good
            sum_all += all
            executable_code += 1
        else:
            broken_code += 1

        if sum_all == 0:
            sum_all = 1
    return sum_good/sum_all, executable_code, broken_code
