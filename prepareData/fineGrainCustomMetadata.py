import json
from pathlib import Path

input_path = Path("../organData/fineGrain/completeDataset/completeDataset.jsonl")
output_path = Path("../organData/fineGrain/completeDataset/test_200.jsonl")

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
                # Only keep 'classification' question if it exists
                original_questions = data.get("questions", {})
                classification_question = {
                    "classification": original_questions["classification"]
                } if "classification" in original_questions else {}
                custom_metadata["questions"] = classification_question
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
