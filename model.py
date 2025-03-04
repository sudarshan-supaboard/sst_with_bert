import torch
import os

from transformers import BertTokenizer, BertForSequenceClassification
from dotenv import load_dotenv
from huggingface_hub import login as hf_login
from wandb import login as wandb_login
from preprocess import idx_to_labels, dataset, labels_to_idx
from peft import  LoraConfig, get_peft_model # type: ignore

from config import Config

load_dotenv()

hf_key = os.environ['HUGGING_FACE_API_KEY']
wandb_key = os.environ['WANDB_API_KEY']


hf_login(token=hf_key)
wandb_login(key=wandb_key)



tokenizer = BertTokenizer.from_pretrained(Config.MODEL_PATH, token=hf_key)
model = BertForSequenceClassification.from_pretrained(Config.MODEL_PATH,
                                                      problem_type="multi_label_classification",
                                                      ignore_mismatched_sizes=True,
                                                      num_labels=len(idx_to_labels),
                                                      torch_dtype=torch.bfloat16
                                                      )

lora_config = LoraConfig(
    r=16,  # Rank of the low-rank matrices (balance between efficiency & expressiveness)
    lora_alpha=32,  # Scaling factor to control update magnitude
    lora_dropout=0.1,  # Dropout to prevent overfitting
    target_modules=["query", "key", "value"],  # Apply LoRA to attention layers only
    bias="none",  # No additional biases for stability
    task_type="SEQ_CLS",  # Sequence classification task
    modules_to_save=['classifier']
)

# # Freeze all layers except the classifier
# for param in model.parameters():
#     param.requires_grad = False

# # Unfreeze the classifier parameters
# for param in model.classifier.parameters():
#     param.requires_grad = True


model = get_peft_model(model=model, peft_config=lora_config)
model.print_trainable_parameters()

# for name, param in model.named_parameters():
#     if param.requires_grad:
#       print(f"Parameter {name}: Trainable={param.requires_grad}")

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
