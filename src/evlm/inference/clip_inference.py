import torch
import torchvision
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from utils  import save_output,init_sub_results
import random


#QUESTIONS = ['modality', 'submodality', 'domain', 'subdomain' , 'stain', 'classification']
def evaluate_dataset( dataset:dict, 
                       model_dict:dict[str,str], 
                       split:str,
                       transform,
                       output_dir,
                       question_key:str = "captions",
                       DEBUG:bool=False) -> None:
    """
    Evaluates a dataset using a given model.

    Parameters:
    dataset (dict): The dataset to be evaluated.
    model_dict (dict[str, str]): Dictionary containing model information.
    split (str): Split of the dataset (e.g., train, test).
    transform: Transformation function for the dataset.
    output_dir: Directory to save the output.
    DEBUG (bool): Whether to run in debug mode. Default is False.

    Returns:
    None
    """
    if question_key=="questions":
        output_name = output_dir / model_dict['name'] / question_key/ dataset["dataset"].name
        results = []
        for j, data_point in enumerate(tqdm( dataset["loader"], desc=f"Evaluating  {dataset['dataset'].name} | model:{model_dict['name']}")):
            if dataset['dataset'].name == "cognition":
                print("setting question key to questions")
                question_key:str = "questions"
            else:
                pass
            print(f"Doing infernece with {question_key}")
            questions:dict[str,str] = data_point['custom_metadata'][question_key]
            # image_id:str = data_point["metadata"]['image_id']
            # if dataset['dataset'].name == "cognition":
            #     image:Path = dataset['dataset'].path / image_id
            # else:
            #     image:Path = dataset['dataset'].path / dataset['dataset'].split / image_id
            image_id:str = data_point["metadata"]['image_id']
            image_filename:str = data_point["metadata"]['image']  # Use the actual image filename
            if dataset['dataset'].name == "cognition":
                image:Path = dataset['dataset'].path / "images" / image_filename
            else:
                # Look for images in the same directory as the JSONL file
                image:Path = dataset['dataset'].path / "images"/ image_filename
            sub_results:dict[str,str] = init_sub_results(data_point)
            
            for question_class in questions.keys():
                q_data = questions.get(question_class)
            
                # Check q_data is a dict (not None/null/str) and has valid structure
                if not isinstance(q_data, dict):
                    print(f"Skipping {image_id} | {question_class} — invalid question block (q_data={q_data})")
                    continue
            
                question = q_data.get("question")
                answer = q_data.get("answer")
                options = q_data.get("options")
            
                # Skip if any are missing or invalid
                if not question or not answer or not options or answer not in options:
                    print(f"Skipping {image_id} | {question_class} — question/answer/options invalid")
                    continue
                result:dict  = {}
                question:str = questions[question_class]["question"]
                answer:str   = questions[question_class]["answer"]
                options:list[str] = questions[question_class]["options"]
                position:int = options.index(answer)
    
                
            
                assert answer in options,f"answer not in options: {answer} not in {options}"
                
                if question_key == "questions":
                    options = [question + " " + option for option in options]
    
                #import pdb; pdb.set_trace()
                # try:
                #     output:dict = model_dict["model"].forward(image,options)
                # except Exception as e:
                #         print(f"Could not run inference for {image_id}, error: {e}")
    
                print(f"image: {image}")
                print(f"type(image): {type(image)}")
                # print(f"text: {options}")
                print(f"type(options): {type(options)}")
                try:
                    #output:dict = model_dict["model"].forward(image,options)
                    output:dict = model_dict["model"].forward([str(image)],options)
    
                except Exception as e:
                    print(f"Could not run inference for {image_id}, error: {e}")
                    # Skip this data point and continue
                    continue
                #import pdb;pdb.set_trace()
    
                result["question_class"] = question_class
                result["questions"]      = options
                result["image_id"]       = image_id
                result["correct_answer"] = answer
                result["correct_idx"]    = position
                result["model_answers"]  = output
    
                
    
                for key in sub_results.keys():
                    result[key] = sub_results[key]
    
    
                results.append(result)
    
            if DEBUG:
                if j == 2:
                    import pdb;pdb.set_trace()
                    break
    elif question_key == "captions":
        output_name = output_dir / model_dict['name'] / question_key/ dataset["dataset"].name
        results = []
        for j, data_point in enumerate(tqdm( dataset["loader"], desc=f"Evaluating  {dataset['dataset'].name} | model:{model_dict['name']}")):
            if dataset['dataset'].name == "cognition":
                print("setting question key to questions")
                question_key:str = "questions"
            else:
                pass
            print(f"Doing infernece with {question_key}")
            questions:dict[str,str] = data_point['custom_metadata'][question_key]
            # image_id:str = data_point["metadata"]['image_id']
            # if dataset['dataset'].name == "cognition":
            #     image:Path = dataset['dataset'].path / image_id
            # else:
            #     image:Path = dataset['dataset'].path / dataset['dataset'].split / image_id
            image_id:str = data_point["metadata"]['image_id']
            image_filename:str = data_point["metadata"]['image']  # Use the actual image filename
            if dataset['dataset'].name == "cognition":
                image:Path = dataset['dataset'].path / "images" / image_filename
            else:
                # Look for images in the same directory as the JSONL file
                image:Path = dataset['dataset'].path / "images"/ image_filename
            sub_results:dict[str,str] = init_sub_results(data_point)
            
            for question_class in questions.keys():
                q_data = questions.get(question_class)
            
                if not isinstance(q_data, dict):
                    print(f"Skipping {image_id} | {question_class} — invalid question block (q_data={q_data})")
                    continue
            
                question = q_data.get("question")
                options = q_data.get("options")
                try:
                    position = int(q_data.get("answer_idx"))
                    answer = options[position]
                except (ValueError, IndexError, TypeError):
                    print(f"Skipping {image_id} | {question_class} — invalid answer index")
                    continue
            
                if not question or not options or not isinstance(options, list) or not answer:
                    print(f"Skipping {image_id} | {question_class} — missing question/options/answer")
                    continue
                print(f"image: {image}")
                try:
                    output:dict = model_dict["model"].forward([str(image)], options)
                except Exception as e:
                    print(f"Could not run inference for {image_id}, error: {e}")
                    continue
            
                result = {
                    "question_class": question_class,
                    "question": question,
                    "questions": options,
                    "image_id": image_id,
                    "correct_answer": answer,
                    "correct_idx": position,
                    "model_answers": output
                }
            
                for key in sub_results:
                    result[key] = sub_results[key]
            
                results.append(result)
    
            if DEBUG:
                if j == 2:
                    import pdb;pdb.set_trace()
                    break

    save_output(results, output_name)


#image:Path   = dataset['dataset'].path /image_id