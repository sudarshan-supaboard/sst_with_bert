import torch

from typing import List
from transformers import BertForSequenceClassification, BertTokenizer
from peft import peft_model, LoraConfig # type: ignore
from config import Config
from pprint import pprint
from dotenv import load_dotenv
from preprocess import idx_to_labels

import os

load_dotenv()

hf_token = os.environ['HUGGING_FACE_API_KEY']

def predict(checkpoint_uri, input: List[str]):
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
    
    model = peft_model.PeftModelForSequenceClassification.from_pretrained(model, checkpoint_uri)
    model.eval()

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
    pred_labels = predict(
        checkpoint_uri="./checkpoints/checkpoint-900",
        input=["Product is good", "Product is bad"],
    )
    
    print(pred_labels)
