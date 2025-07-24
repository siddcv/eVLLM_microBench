import glob
import os
import json
from pathlib import Path
from random import shuffle

from torch.utils.data import Dataset,DataLoader

def check_json_file(file_path):
    if not os.path.exists(file_path):
        print("File does not exist.")
        return
    
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            print("Error decoding JSON.")
            return
        
        if data:  # Checks if the data is not empty
            return True

        elif data == {}:
            return False

        else:
            return False
            
def read_jsonl(file_path):
    """
    Read data from a JSONL (JSON Lines) file.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        list: List of dictionaries, each containing data from one line of the JSONL file.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    return data

class JsonlDataset(Dataset):
    """
    Dataset class to load JSON files.

    Args:
        dataset_path (str | Path): Path to the dataset directory.
        extension (str, optional): File extension of the JSON files. Defaults to "json".
        limit (int, optional): Maximum number of files to load. Defaults to None.
        shuffle_dataset (bool, optional): Whether to shuffle the dataset. Defaults to True.
    """

    def __init__(self, dataset_path: str | Path, 
                       split: str = None,
                       extension: str = "json", 
                       limit: int = None, 
                       shuffle_dataset: bool = False,
                       verbose:bool=True):

        
        if isinstance(dataset_path,str):
            dataset_path: Pat = Path(dataset_path)
            
    
        self.path: Path     = dataset_path
        self.name:str       =  dataset_path.name
        self.split:str      = split
        self.extension: str = extension
        self.limit: int = limit
        self.shuffle_dataset: bool = shuffle_dataset
        self.verbose:bool = verbose
        self.files: list[str] = self.get_files()

    
        #import pdb;pdb.set_trace()
    def get_name():
        return self.name

    def get_files(self):
        # Try to use test_200.jsonl, otherwise use the first .jsonl file found
        data_path = os.path.join(self.path, "test_200.jsonl")
        if not os.path.exists(data_path):
            jsonl_files = list(Path(self.path).glob("*.jsonl"))
            if not jsonl_files:
                raise FileNotFoundError(f"No .jsonl file found in {self.path}")
            data_path = str(jsonl_files[0])
            print(f"[JsonlDataset] test_200.jsonl not found, using {data_path}")
        else:
            print(f"[JsonlDataset] Using {data_path}")
        data = read_jsonl(data_path)
        print(f"Loaded {len(data)} datapoints from {data_path}")

        if self.shuffle_dataset:
            shuffle(data)

        if self.limit is not None:
            data = data[:self.limit]

        print("="*80)
        print(f"Dataset {self.path} initialized with {len(data)} datapoints")
        print("="*80)

        return data

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx]

def collate_fn(batch):
    """
    Custom collate function to handle variable length data in the batch.
    """
    return batch


if __name__ == "__main__":
    split: str = 'validation'
    DATA_ROOT = Path("/pasteur/data/jnirschl/datasets/biovlmdata/data/processed/")
    sub_datasets: list[str] = os.listdir(DATA_ROOT)
    sub_datasets = list(set(sub_datasets) - set(['bravura.tex', 'bravura.md','image_json_pairs.jsonl', 'bravura.pdf', 'bravura.feather',"burgess_et_al_2024"]))

    dataset_path = DATA_ROOT / sub_datasets[0] 
    print(dataset_path)
    custom_dataset = JsonlDataset(dataset_path,split=split, limit=1000)
    data_loader    = DataLoader(custom_dataset, batch_size=2, collate_fn=collate_fn)

    # Iterate over the data loader
    for data_point in custom_dataset:
        print(data_point)  # Here 'batch' will contain a list of JSON data from the files
        import pdb;pdb.set_trace()
        
