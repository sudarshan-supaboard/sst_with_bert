import torch

from typing import List
from transformers import BertForSequenceClassification, BertTokenizer
from peft import peft_model
from config import Config
from pprint import pprint
from dotenv import load_dotenv
from preprocess import idx_to_labels

import os
import pandas as pd

load_dotenv()

hf_token = os.environ['HUGGING_FACE_API_KEY']

model = BertForSequenceClassification.from_pretrained(
        Config.MODEL_PATH,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
        num_labels=Config.NUM_LABELS,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        
    )
    
tokenizer = BertTokenizer.from_pretrained(
        Config.MODEL_PATH,
        token=hf_token
)
    
model = peft_model.PeftModelForSequenceClassification.from_pretrained(model, "./checkpoints/checkpoint-10200")
model.eval()

def predict(input: List[str]):
    
    input_tokens = tokenizer(
        input, padding="max_length", truncation=True, return_tensors="pt"
    )
    
    

    with torch.no_grad():
        preds = model(input_tokens['input_ids'], input_tokens['attention_mask'])
        
        logits = torch.softmax(preds.logits, dim=1)
        
        pred_labels = []
        for i in torch.argmax(logits,dim=1):
            pred_labels.append(idx_to_labels[i])
    
    return pred_labels
    
    
if __name__ == "__main__":
    

    df = pd.read_csv("/traditional_wear_test.csv")

    reviews = df['review'].to_list()
    ratings = df['rating'].to_list()
    
    preds =[]
    for i in range(0,len(reviews), 8):
        print(f'{i}-{i+8}/{len(reviews)}')
        preds_ = predict(input=reviews[i: i+8])
        preds.extend(preds_)
        if i % 64 == 0:
            confmat= pd.crosstab(index=preds,colnames=ratings[:i+8],normalize='index')*100 # type: ignore
            confmat.to_csv('confmat_trad.csv')

    
    confmat= pd.crosstab(index=preds,colnames=ratings[:i+8],normalize='index')*100 # type: ignore
    confmat.to_csv('confmat_trad.csv')
