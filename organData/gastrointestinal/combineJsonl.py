from pathlib import Path

# Define paths
base_path = Path("/workspace/eVLLM_Sidd/eVLLM_microBench/organData/gastrointestinal")
file1 = base_path / "test_100.jsonl"
file2 = base_path / "test_150.jsonl"
output_file = base_path / "test_200.jsonl"

# Combine contents
with output_file.open("w") as fout:
    for file in [file1, file2]:
        with file.open("r") as fin:
            for line in fin:
                fout.write(line)

print(f"Combined {file1.name} and {file2.name} into {output_file.name}")