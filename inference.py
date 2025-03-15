import torch

from typing import List
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from peft import peft_model
from config import Config
from pprint import pprint
from dotenv import load_dotenv
from preprocess import idx_to_labels
from pathlib import Path
import os
import pandas as pd
import kagglehub

load_dotenv()

def download_data():
    # Download latest version
    path = kagglehub.dataset_download("sudarshan1927/reviews")

    print("Path to dataset files:", path)
    return path

def get_model(model_name: str, checkpoint: str):
    model = None
    tokenizer = None

    if model_name == "bert":
        Config.set_bert_model()
        print(f"Model: {model_name} | Model Path: {Config.MODEL_PATH}")
        print(f"Output Dir: {Config.OUTPUT_DIR}")

        tokenizer = BertTokenizer.from_pretrained(Config.MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(
            Config.MODEL_PATH,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
            num_labels=Config.NUM_LABELS,
            torch_dtype=torch.bfloat16,
            force_download=False,
            local_files_only=True
        )
    elif model_name == "roberta":
        Config.set_roberta_model()
        print(f"Model: {model_name} | Model Path: {Config.MODEL_PATH}")
        print(f"Output Dir: {Config.OUTPUT_DIR}")

        tokenizer = RobertaTokenizer.from_pretrained(Config.MODEL_PATH, force_download=True)
        model = RobertaForSequenceClassification.from_pretrained(
            Config.MODEL_PATH,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
            num_labels=Config.NUM_LABELS,
            torch_dtype=torch.bfloat16,
            force_download=False,
            local_files_only=True
        )

    else:
        raise ValueError("Invalid model name")

    model = peft_model.PeftModelForSequenceClassification.from_pretrained(model, f"./{Config.OUTPUT_DIR}/{checkpoint}")
    model.eval()

    return model, tokenizer


def predict(model, tokenizer, input: List[str], device):
    
    
    inputs = tokenizer(
        input, padding="max_length", truncation=True, return_tensors="pt", max_length=512
    )
    
    inputs = {k: v.to(device) for k, v in inputs}

    with torch.no_grad():
        preds = model(**inputs)
        
        logits = torch.softmax(preds.logits.cpu(), dim=1)
        
        pred_labels = []
        for i in torch.argmax(logits,dim=1):
            pred_labels.append(idx_to_labels[i])
            
    inputs = {k: v.to("cpu") for k, v in inputs}
    
    
    return pred_labels
    
    
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ds_path = download_data()
    ds_path = Path(ds_path)
    
    model, tokenizer = get_model("roberta", "checkpoint-4900")
    model.to(device)
    
    ds_path = ds_path / 'toy_store_test.csv'
    df = pd.read_csv(ds_path)

    reviews = df['review'].to_list()
    ratings = df['rating'].to_list()
    
    preds = []
    for i in range(0,len(reviews), 8):
        print(f'{i}-{i+8}/{len(reviews)}')
        preds_ = predict(model, tokenizer, input=reviews[i: i+8], device=device)
        preds.extend(preds_)
        if i % 64 == 0:
            confmat= pd.crosstab(index=preds,colnames=ratings[:i+8],normalize='index')*100 # type: ignore
            confmat.to_csv('confmat_toy.csv')

    model.cpu()
    confmat= pd.crosstab(index=preds,colnames=ratings[:i+8],normalize='index')*100 # type: ignore
    confmat.to_csv('confmat_toy.csv')
