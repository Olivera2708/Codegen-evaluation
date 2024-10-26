import subprocess
import os

def run_tests(generated_code, test_cases):
    with open('temp_code.kt', 'w', encoding='utf-8') as f:
        f.write(generated_code)

    with open('temp_code.kt', 'a', encoding='utf-8') as f:
        f.write(test_cases)

    try:
        subprocess.run(['C:\\Program Files\\kotlin\\bin\\kotlinc.bat', 'temp_code.kt', '-include-runtime', '-d', 'temp_code.jar'], check=True)
        output = subprocess.run(['java', '-jar', 'temp_code.jar'], capture_output=True, text=True)
        output_lines = output.stdout.strip().split('\n')
        all = len(output_lines)
        good = all - sum("Exception" in line for line in output_lines)
        return True, good, all
        
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e.stderr}")
        return False, 0, 0
    
    finally:
        if os.path.exists('temp_code.kt'):
            os.remove('temp_code.kt')
        if os.path.exists('temp_code.jar'):
            os.remove('temp_code.jar')
