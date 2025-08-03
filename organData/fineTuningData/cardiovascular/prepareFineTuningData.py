# import json

# def resolve_caption_task(name: str) -> str:
#     # Define which names are considered coarse or fine grained
#     coarse_keys = {"modality", "submodality", "stain", "domain", "subdomain"}
#     fine_keys = {"classification"}

#     if name in coarse_keys:
#         return "coarse_caption"
#     elif name in fine_keys:
#         return "fine_caption"
#     else:
#         return "unknown_caption"

# def process_jsonl(input_path, output_path):
#     processed = []

#     with open(input_path, "r") as infile:
#         for line in infile:
#             entry = json.loads(line.strip())

#             image_id = entry["image_id"]
#             image_path = f"images/{entry['image']}"  # assumes images/<image_name>.png format

#             # Process each caption
#             for caption_key, caption_data in entry.get("captions", {}).items():
#                 name = caption_data["name"]  # e.g., "modality", "classification"
#                 options = caption_data["options"]
#                 answer_idx = int(caption_data["answer_idx"])
#                 caption_text = options[answer_idx]

#                 processed.append({
#                     "image_id": image_id,
#                     "image_path": image_path,
#                     "text": caption_text,
#                     "task": resolve_caption_task(name),
#                     "label": caption_text,
#                     "caption": caption_key
#                 })

#     with open(output_path, "w") as outfile:
#         for item in processed:
#             outfile.write(json.dumps(item) + "\n")

# # Example usage:
# process_jsonl("cardiovascular.jsonl", "fineTuningData.jsonl")




# import json
# import re

# def extract_task(caption_key):
#     # Extracts the task name like "modality", "stain", "domain", etc. from "modality_0"
#     return caption_key.split("_")[0]

# def process_jsonl(input_file_path, output_file_path):
#     output_data = []

#     with open(input_file_path, 'r') as f:
#         for line in f:
#             data = json.loads(line)

#             image_id = data["image_id"]
#             image_path = f"images/{data['image']}"  # assumes images/<image_name>.png format
#             caption_data = data["captions"]

#             for caption_key, caption_data in data.get("captions", {}).items():
#                 name = caption_data["name"]  # e.g., "modality", "classification"
#                 options = caption_data["options"]
#                 answer_idx = int(caption_data["answer_idx"])
#                 task_name = extract_task(caption_key)
#                 for idx, option in enumerate(options):
#                     caption_text = options[idx]
#                     output_data.append({
#                         "image_id": image_id,
#                         "image_path": image_path,
#                         "text": caption_text,
#                         "is_positive": idx == answer_idx,
#                         "task": task_name,
#                         "caption": caption_key
#                     })

#     with open(output_file_path, 'w') as out_f:
#         for item in output_data:
#             out_f.write(json.dumps(item) + "\n")

#     print(f"Processed {len(output_data)} entries from {input_file_path} into {output_file_path}")

# # Run once for all tasks
# process_jsonl("cardiovascular.jsonl", "fineTuningData.jsonl")





import json
import re

def extract_task(caption_key):
    # Extracts the task name like "modality", "stain", "domain", etc. from "modality_0"
    return caption_key.split("_")[0]

def resolve_caption_task(name: str) -> str:
    # Define which names are considered coarse or fine grained
    coarse_keys = {"modality", "submodality", "stain", "domain", "subdomain"}
    fine_keys = {"classification"}

    if name in coarse_keys:
        return "coarse_caption"
    elif name in fine_keys:
        return "fine_caption"
    else:
        return "unknown_caption"

# def process_jsonl(input_file_path, output_file_path):
#     output_data = []

#     with open(input_file_path, 'r') as f:
#         for line in f:
#             data = json.loads(line)

#             image_id = data["image_id"]
#             # workspace/eVLLM_Sidd/eVLLM_microBench/organData/fineTuningData/cardiovascular/images
#             image_path = f"/workspace/eVLLM_Sidd/eVLLM_microBench/organData/fineTuningData/cardiovascular/images/{data['image']}"  # assumes images/<image_name>.png format
#             caption_data = data["captions"]

#             for caption_key, caption_data in data.get("captions", {}).items():
#                 name = caption_data["name"]  # e.g., "modality", "classification"
#                 options = caption_data["options"]
#                 answer_idx = int(caption_data["answer_idx"])
#                 task_name = extract_task(caption_key)
#                 for idx, option in enumerate(options):
#                     caption_text = options[idx]
#                     output_data.append({
#                         "image_id": image_id,
#                         "image_path": image_path,
#                         "text": caption_text,
#                         "is_positive": idx == answer_idx,
#                         "task": resolve_caption_task(name),
#                         "caption": caption_key
#                     })

#     with open(output_file_path, 'w') as out_f:
#         for item in output_data:
#             out_f.write(json.dumps(item) + "\n")

def process_jsonl(input_file_path, output_file_path):
    import json
    output_data = []

    with open(input_file_path, 'r') as f:
        for line in f:
            data = json.loads(line)

            image_id = data["image_id"]
            image_path = f"/workspace/eVLLM_Sidd/eVLLM_microBench/organData/fineTuningData/cardiovascular/images/{data['image']}"
            
            for caption_key, caption_data in data.get("captions", {}).items():
                name = caption_data["name"]
                options = caption_data["options"]
                answer_idx = int(caption_data["answer_idx"])
                task_name = resolve_caption_task(name)

                # Collect positives and negatives
                positive = {
                    "image_id": image_id,
                    "image_path": image_path,
                    "text": options[answer_idx],
                    "is_positive": True,
                    "task": task_name,
                    "caption": caption_key
                }

                negatives = []
                for idx, option in enumerate(options):
                    if idx != answer_idx:
                        negatives.append({
                            "image_id": image_id,
                            "image_path": image_path,
                            "text": option,
                            "is_positive": False,
                            "task": task_name,
                            "caption": caption_key
                        })

                # Add positive and up to 2 negatives
                output_data.append(positive)
                output_data.extend(negatives[:1])  # At most 2 negatives

    # Save to output file
    with open(output_file_path, 'w') as out_f:
        for item in output_data:
            out_f.write(json.dumps(item) + '\n')


    print(f"Processed {len(output_data)} entries from {input_file_path} into {output_file_path}")

# Run once for all tasks
process_jsonl("cardiovascular.jsonl", "fineTuningData1.jsonl")