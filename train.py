import torch
import numpy as np
import torch.nn.functional as F
import evaluate
import json
import argparse
import wandb

from transformers import Trainer
from transformers import Trainer, TrainingArguments

from pprint import pprint
from utils import EarlyStoppingTrainingLossCallback, GCSUploadCallback, get_memory_usage
from model import model, tokenized_datasets
from config import Config

# Load multiple metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

wandb.init(project=Config.PROJECT_NAME)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],  # type: ignore
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],  # type: ignore
    }


class CustomTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        
        device = 'cpu'
        
        if isinstance(model, torch.nn.DataParallel):
            device = model.module.device
        
        
        labels = inputs.pop("labels")
        labels = labels.to(device)

        # Forward pass
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits

        # Example: Custom loss (Focal Loss)
        loss = F.cross_entropy(logits, labels)

        # get_memory_usage()
        return (loss, outputs) if return_outputs else loss


def train(bkt_upload=True):

    # Create TrainingArguments
    training_args = TrainingArguments(
        run_name=Config.PROJECT_NAME,
        output_dir=Config.OUTPUT_DIR,  # Output directory
        num_train_epochs=5,  # Total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=64,  # Batch size for evaluation
        warmup_ratio=0.1,  # Number of warmup steps for learning rate scheduler
        learning_rate=5e-5,
        weight_decay=0.01,  # Strength of weight decay
        logging_dir="./logs",  # Directory for storing logs
        logging_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=4,
        save_steps=300,
        eval_steps=300,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb",
        bf16=True,
    )


    es_callback = EarlyStoppingTrainingLossCallback(patience=3)
    gcs_callback = GCSUploadCallback(bkt_upload=bkt_upload)

    # Create Trainer instance
    trainer = CustomTrainer(
        model=model,  # The instantiated 🤗 Transformers model to be trained
        args=training_args,  # Training arguments, defined above
        train_dataset=tokenized_datasets["train"],  # Training dataset
        eval_dataset=tokenized_datasets["valid"],  # Evaluation dataset
        compute_metrics=compute_metrics,
        callbacks=[es_callback, gcs_callback],  # Stop if no improvement in 3 evaluations
    )

    # Train the model
    trainer.train()

    prediction_outputs = trainer.predict(test_dataset=tokenized_datasets["test"])  # type: ignore
    best_checkpoint = trainer.state.best_model_checkpoint

    print("predictions")
    pprint(prediction_outputs.metrics)
    
    with open("best_checkpoint.json", "w") as f:
        json.dump(obj={"checkpoint": best_checkpoint}, fp=f)
    print(f"Best Checkpoint: {best_checkpoint} Saved.")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="User Info CLI")
    parser.add_argument("-u", "--upload", action="store_true", help="Enable uploads")
    args = parser.parse_args()
    
    bkt_upload = False
    if args.upload:
        print(f'bucket upload enabled')
        bkt_upload=True
    else:
        print(f'bucket upload disabled')
    
    train(bkt_upload=bkt_upload)


