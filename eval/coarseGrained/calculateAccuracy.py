import pandas as pd
import ast
import re
import math
# Load the CSV
df = pd.read_csv("/workspace/eVLLM_Ayman/output_results/coarse_grained_tasks/BioMedCLIP_Base/completeDataset/0.70/completeDataset.csv")
# workspace/eVLLM_Sidd/eVLLM_microBench/output_results/coarse_grained_tasks/BioMedCLIP/questions/hematopathology.csv

right_answers=0
questions=0
# Process first 20 rows
for idx, row in df.iterrows():
    correct_answer = row["correct_answer"]
    model_answers_raw = row["model_answers"]
    questions=questions+1
    # print(f"Row {idx + 1}:")
    # print(f"  correct_answer: {correct_answer}")
    # print(f"  type(model_answers): {type(model_answers_raw)}")
    # print(f"  (model_answers): {model_answers_raw}")

    try:
        # Parse the model_answers string safely
        model_answers = ast.literal_eval(model_answers_raw)
        pred_prompt_list = model_answers.get("pred_prompt", [])

        if pred_prompt_list and isinstance(pred_prompt_list[0], str):
            match = re.search(r'\?(.*)', pred_prompt_list[0])
            if match:
                predicted_ans = match.group(1).strip()
                # print(f"  after '?': {predicted_ans}")
            else:
                print("  after '?': <no match>")
        else:
            print("  pred_prompt missing or malformed")
    except Exception as e:
        print(f"  Error parsing model_answers: {e}")

    if(correct_answer == predicted_ans):
        right_answers=right_answers+1

    # print("-" * 60)
accuracy_prop = right_answers / questions  # value between 0 and 1

se = math.sqrt((accuracy_prop * (1 - accuracy_prop)) / questions)
ci_range = 1.96 * se  # for 95% CI

lower_ci = (accuracy_prop - ci_range) * 100
upper_ci = (accuracy_prop + ci_range) * 100
accuracy=accuracy_prop*100
print("completeDataset")
print(f"Number of questions: {questions}")
print(f"Number of correct answers: {right_answers}")
print(f"Accuracy: {accuracy:.3f}%")
print(f"95% Confidence Interval: ({lower_ci:.3f}%, {upper_ci:.3f}%)")
