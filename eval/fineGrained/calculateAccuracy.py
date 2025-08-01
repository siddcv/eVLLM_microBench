import pandas as pd
import ast
import re
import math

# Load the CSV
df = pd.read_csv("../../output_results/fine_grained_tasks/BioMedCLIP/questions/completeDataset.csv")

right_answers = 0
questions = 0

for idx, row in df.iterrows():
    correct_answer = row["correct_answer"]
    model_answers_raw = row["model_answers"]
    predicted_ans = None  # Initialize to avoid undefined error
#######################change for fine grained hematopathology##########################
    try:
        model_answers = ast.literal_eval(model_answers_raw)
        pred_prompt_list = model_answers.get("pred_prompt", [])

        if pred_prompt_list and isinstance(pred_prompt_list[0], str):
            match = re.search(r'\?(.*)', pred_prompt_list[0])
            if match:
                predicted_ans = match.group(1).strip()
            else:
                match = re.search(r':(.*)', pred_prompt_list[0])
                if match:
                    predicted_ans = match.group(1).strip()
                else:
                    print(f"Row {idx + 1}: No match after '?'")
        else:
            print(f"Row {idx + 1}: pred_prompt missing or malformed")
    except Exception as e:
        print(f"Row {idx + 1}: Error parsing model_answers: {e}")

    if predicted_ans is not None:
        questions += 1
        if correct_answer == predicted_ans:
            right_answers += 1
    else:
        print(f"Row {idx + 1}: Skipping due to missing predicted_ans")

accuracy_prop = right_answers / questions if questions > 0 else 0
se = math.sqrt((accuracy_prop * (1 - accuracy_prop)) / questions) if questions > 0 else 0
ci_range = 1.96 * se

lower_ci = (accuracy_prop - ci_range) * 100
upper_ci = (accuracy_prop + ci_range) * 100
accuracy = accuracy_prop * 100

print("completeDataset")
print(f"Number of questions used: {questions}")
print(f"Number of correct answers: {right_answers}")
print(f"Accuracy: {accuracy:.3f}%")
print(f"95% Confidence Interval: ({lower_ci:.3f}%, {upper_ci:.3f}%)")
