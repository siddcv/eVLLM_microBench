from datasets import load_dataset
import json

# Load dataset
dataset_dict = load_dataset("teamasap/cardiovascular-pathology-dataset")
dataset = dataset_dict["test"]  # or "train" if exists

# List all metadata fields you want to extract
metadata_fields = [
    "image_id",
    "image",  # Will convert below
    "label",
    "label_name",
    "dataset",
    "domain",
    "institution",
    "license",
    "microns_per_pixel",
    "modality",
    "ncbitaxon_id",
    "ncbitaxon_name",
    "pmid",
    "split",
    "stain",
    "subdomain",
    "submodality",
    "synthetic",
]

# output_file = "/workspace/organData/cardiovascular/cardiovascular.jsonl"
output_file="/workspace/eVLLM_Sidd/eVLLM_microBench/organData/cardiovascular/cardiovascular.jsonl"

with open(output_file, "w") as fout:
    for row in dataset:
        output = {}

        # Add metadata fields
        for field in metadata_fields:
            val = row.get(field)
            if field == "image":
                # Convert PIL image object to filename string or use image_id + ".png"
                val = row.get("image_id", "unknown") + ".png"
            output[field] = val

        # Add full captions dict (all keys)
        if "captions" in row and isinstance(row["captions"], dict):
            output["captions"] = row["captions"]

        # Add full questions dict (all keys)
        if "questions" in row and isinstance(row["questions"], dict):
            output["questions"] = row["questions"]

        # Write the JSON line
        fout.write(json.dumps(output) + "\n")

print(f"Done! Output saved to {output_file}")
print(dataset[0].keys())             # e.g., dict_keys(['image', 'captions', 'questions'])
print(json.dumps(dataset[0]['questions'], indent=2))  # pretty print nested questions

