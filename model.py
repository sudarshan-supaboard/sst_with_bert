from typing import Dict, List
import torch
import os

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from preprocess import make_datasets
from peft import LoraConfig, get_peft_model  # type: ignore

from config import Config


def get_model(model_name: str):
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
            torch_dtype=torch.float16,
        )
    elif model_name == "roberta":
        Config.set_roberta_model()
        print(f"Model: {model_name} | Model Path: {Config.MODEL_PATH}")
        print(f"Output Dir: {Config.OUTPUT_DIR}")

        tokenizer = RobertaTokenizer.from_pretrained(Config.MODEL_PATH)
        model = RobertaForSequenceClassification.from_pretrained(
            Config.MODEL_PATH,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
            num_labels=Config.NUM_LABELS,
            torch_dtype=torch.float16,
        )

    else:
        raise ValueError("Invalid model name")

    lora_config = LoraConfig(
        r=16,  # Rank of the low-rank matrices (balance between efficiency & expressiveness)
        lora_alpha=32,  # Scaling factor to control update magnitude
        lora_dropout=0.1,  # Dropout to prevent overfitting
        target_modules=["query", "key", "value"],  # Apply LoRA to attention layers only
        bias="none",  # No additional biases for stability
        task_type="SEQ_CLS",  # Sequence classification task
        modules_to_save=["classifier"],
    )

    model = get_peft_model(model=model, peft_config=lora_config)
    model.print_trainable_parameters()

    return model, tokenizer



def tokenize_function(examples: Dict[str, List[str]], tokenizer: BertTokenizer | RobertaTokenizer):

    batch_inputs = tokenizer(
        examples["text"], padding="max_length", truncation=True, return_tensors="pt", max_length=512
    )

    # convert labels to ids using labels_to_idx
    batch_inputs["label"] = torch.tensor(examples["label"])

    return batch_inputs

def tokenize(tokenizer):
    dataset = make_datasets()
    
    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer}, remove_columns=['text']
    )

    tokenized_datasets.set_format(  # type: ignore
        type="torch",
        # columns=["input_ids","token_type_ids", "attention_mask", "label"],
        device="cpu",
        # output_all_columns=False,
    )

    return tokenized_datasets

if __name__ == '__main__':
    model, tokenizer = get_model('bert')
    
    tokenized_datasets = tokenize(tokenizer)
    print(tokenized_datasets['train'][0])
