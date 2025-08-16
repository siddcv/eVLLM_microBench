# import json
# from pathlib import Path

# input_path = Path("../organData/cardiovascular/cardiovascular.jsonl")
# output_path = Path("../organData/cardiovascular/test_200.jsonl")

# with input_path.open("r") as fin, output_path.open("w") as fout:
#     for line in fin:
#         data = json.loads(line)
#         # Wrap "questions" inside "custom_metadata"
#         questions = data.pop("questions", None)
#         if questions is not None:
#             data["custom_metadata"] = {"questions": questions}
#         else:
#             # if no questions found, keep as is or add empty dict
#             data["custom_metadata"] = {"questions": {}}
#         # Write back as JSONL line
#         fout.write(json.dumps(data) + "\n")

# print(f"Finished writing updated file to {output_path}")




# import json
# from pathlib import Path

# input_path = Path("../organData/cardiovascular/cardiovascular.jsonl")
# output_path = Path("../organData/cardiovascular/test_200.jsonl")

# with input_path.open("r") as fin, output_path.open("w") as fout:
#     for line in fin:
#         data = json.loads(line)

#         # Extract "questions" and "captions"
#         questions = data.pop("questions", None)
#         captions = data.pop("captions", None)

#         # Build custom_metadata dict
#         metadata = {}
#         if questions is not None:
#             metadata["questions"] = questions
#         else:
#             metadata["questions"] = {}

#         if captions is not None:
#             metadata["captions"] = captions
#         else:
#             metadata["captions"] = []

#         # Add to data
#         data["custom_metadata"] = metadata

#         # Write updated data
#         fout.write(json.dumps(data) + "\n")

# print(f"Finished writing updated file to {output_path}")





# import json
# from pathlib import Path

# input_path = Path("../organData/cardiovascular/cardiovascular.jsonl")
# output_path = Path("../organData/gastrointestinal/test_200.jsonl")

# with input_path.open("r") as fin, output_path.open("w") as fout:
#     for line in fin:
#         data = json.loads(line)

#         # Extract "questions" and "captions"
#         questions = data.pop("questions", {})
#         captions = data.pop("captions", {})

#         # Everything else goes into "metadata"
#         metadata_fields = {
#             k: data.pop(k) for k in list(data.keys())
#             if k not in ["custom_metadata"]
#         }

#         # Construct final structure
#         data["metadata"] = metadata_fields
#         data["custom_metadata"] = {
#             "questions": questions,
#             "captions": captions
#         }

#         fout.write(json.dumps(data) + "\n")

# print(f"Finished writing updated file to {output_path}")




# import json
# from pathlib import Path

# input_path = Path("../organData/coarseGrain/cardiovascular/cardiovascular.jsonl")
# output_path = Path("../organData/coarseGrain/cardiovascular/test_200.jsonl")

# custom_fields = [
#     "questions", "captions", "microns_per_pixel",
#     "domain", "subdomain", "modality", "submodality"
# ]

# with input_path.open("r") as fin, output_path.open("w") as fout:
#     for line in fin:
#         data = json.loads(line)

#         # Extract desired custom metadata fields
#         custom_metadata = {
#             key: data.pop(key, {}) for key in custom_fields
#         }

#         # Remaining fields become metadata
#         metadata_fields = {
#             k: data.pop(k) for k in list(data.keys())
#             if k not in ["custom_metadata"]
#         }

#         # Construct final structure
#         data["metadata"] = metadata_fields
#         data["custom_metadata"] = custom_metadata

#         fout.write(json.dumps(data) + "\n")

# print(f"Finished writing updated file to {output_path}")





import json
from pathlib import Path

input_path = Path("../organData/coarseGrain/completeDataset/completeDataset.jsonl")
output_path = Path("../organData/coarseGrain/completeDataset/test_200.jsonl")

custom_fields = [
    "questions", "captions", "microns_per_pixel",
    "domain", "subdomain", "modality", "submodality"
]

with input_path.open("r") as fin, output_path.open("w") as fout:
    for line in fin:
        data = json.loads(line)

        custom_metadata = {}

        for key in custom_fields:
            if key == "questions":
                # Filter out 'classification' from the questions
                original_questions = data.get("questions", {})
                filtered_questions = {
                    k: v for k, v in original_questions.items() if k != "classification"
                }
                custom_metadata["questions"] = filtered_questions
                data.pop("questions", None)  # Remove from top-level
            else:
                custom_metadata[key] = data.pop(key, {})

        # Remaining fields become metadata
        metadata_fields = {
            k: data.pop(k) for k in list(data.keys())
            if k != "custom_metadata"
        }

        # Construct final structure
        data["metadata"] = metadata_fields
        data["custom_metadata"] = custom_metadata

        fout.write(json.dumps(data) + "\n")

print(f"Finished writing updated file to {output_path}")

