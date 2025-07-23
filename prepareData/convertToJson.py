# import json
# from pathlib import Path

# jsonl_path = Path("/workspace/eVLLM_Sidd/eVLLM_microBench/organData/cardiovascular/cardiovascular.jsonl")
# output_dir=Path("/workspace/eVLLM_Sidd/eVLLM_microBench/organData/cardiovascular/json")
# # output_dir = jsonl_path.parent / "json"
# output_dir.mkdir(parents=True, exist_ok=True)

# with open(jsonl_path, "r") as fin:
#     for i, line in enumerate(fin):
#         data = json.loads(line)
#         image_id = data.get("image_id", f"sample_{i}")
#         out_path = output_dir / f"{image_id}.json"
#         with open(out_path, "w") as fout:
#             json.dump(data, fout)


import json
from pathlib import Path

jsonl_path = Path("/workspace/eVLLM_Sidd/eVLLM_microBench/organData/cardiovascular/cardiovascular.jsonl")
output_dir = Path("/workspace/eVLLM_Sidd/eVLLM_microBench/organData/cardiovascular/json")
output_dir.mkdir(parents=True, exist_ok=True)

for_json_images = "images"  # folder name for images

with open(jsonl_path, "r") as fin:
    for i, line in enumerate(fin):
        data = json.loads(line)
        image_id = data.get("image_id", f"sample_{i}")
        # Update the image field to include the relative path
        data["image"] = f"{for_json_images}/{image_id}.png"
        out_path = output_dir / f"{image_id}.json"
        with open(out_path, "w") as fout:
            json.dump(data, fout)