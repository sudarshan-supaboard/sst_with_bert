import torch
import os

from transformers import BertTokenizer, BertForSequenceClassification
from dotenv import load_dotenv

from preprocess import idx_to_labels, dataset, labels_to_idx
from utils import get_device

from huggingface_hub import login as hf_login
from wandb import login as wandb_login

load_dotenv()

hf_key = os.environ['HUGGING_FACE_API_KEY']
wandb_key = os.environ['WANDB_API_KEY']


hf_login(token=hf_key)
wandb_login(key=wandb_key)



model_path = "nlptown/bert-base-multilingual-uncased-sentiment"

tokenizer = BertTokenizer.from_pretrained(model_path, token=hf_key)
model = BertForSequenceClassification.from_pretrained(model_path,
                                                      problem_type="multi_label_classification",
                                                      ignore_mismatched_sizes=True,
                                                      num_labels=len(idx_to_labels),
                                                      device_map="auto",
                                                      torch_dtype=torch.bfloat16
                                                      )


# Freeze all layers except the classifier
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the classifier parameters
for param in model.classifier.parameters():
    param.requires_grad = True

for name, param in model.named_parameters():
    if param.requires_grad:
      print(f"Parameter {name}: Trainable={param.requires_grad}")

def tokenize_function(examples):

    batch_inputs = tokenizer(examples["text"], padding="max_length",
                             truncation=True, return_tensors="pt")

    # convert labels to ids using labels_to_idx
    batch_inputs['label'] = torch.tensor([labels_to_idx[l] for l in examples['label']])
    # batch_inputs["label"] = torch.tensor(encoder.transform(np.array(examples['label']).reshape(-1,1)))

    return batch_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets.set_format(type='torch',
                              columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'],
                              device='cpu',
                              output_all_columns=False)
