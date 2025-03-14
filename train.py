import torch
import numpy as np
import torch.nn.functional as F
import evaluate
import json
import argparse
import wandb
import os

from transformers import Trainer
from transformers import Trainer, TrainingArguments

from pprint import pprint
from utils import EarlyStoppingTrainingLossCallback, GCSUploadCallback
from model import get_model, tokenize
from config import Config
from dotenv import load_dotenv

from huggingface_hub import login as hf_login
from wandb import login as wandb_login

# Load multiple metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

wandb.init(project=Config.PROJECT_NAME)

load_dotenv()

hf_key = os.environ['HUGGING_FACE_API_KEY']
wandb_key = os.environ['WANDB_API_KEY']


hf_login(token=hf_key)
wandb_login(key=wandb_key)

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

        labels = inputs.pop("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Example: Custom loss (Focal Loss)
        loss = F.cross_entropy(logits, labels)

        # get_memory_usage()
        return (loss, outputs) if return_outputs else loss


def train(
    model,
    bkt_upload,
    num_epochs,
    resume_checkpoint,
    train_batch,
    eval_batch,
    grad_steps,
    log_steps,
    save_steps,
    eval_steps,
):

    model, tokenizer = get_model(model_name=model)
    # Create TrainingArguments
    training_args = TrainingArguments(
        run_name=Config.PROJECT_NAME,
        output_dir=Config.OUTPUT_DIR,  # Output directory
        num_train_epochs=num_epochs,  # Total number of training epochs
        per_device_train_batch_size=train_batch,  # batch size per device during training
        gradient_accumulation_steps=grad_steps,
        per_device_eval_batch_size=eval_batch,  # Batch size for evaluation
        warmup_ratio=0.1,  # Number of warmup steps for learning rate scheduler
        learning_rate=5e-5,
        weight_decay=0.01,  # Strength of weight decay
        logging_dir="./logs",  # Directory for storing logs
        logging_steps=log_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=4,
        save_steps=save_steps,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb",
        fp16=True,
        ddp_find_unused_parameters=False,
        ddp_backend='nccl'
    )

    es_callback = EarlyStoppingTrainingLossCallback(patience=3)
    gcs_callback = GCSUploadCallback(bucket_name=Config.BUCKET_NAME, checkpoint_dir=Config.OUTPUT_DIR, bkt_upload=bkt_upload)

    tokenized_datasets = tokenize(tokenizer)
    # Create Trainer instance
    trainer = CustomTrainer(
        model=model,  # The instantiated 🤗 Transformers model to be trained
        args=training_args,  # Training arguments, defined above
        train_dataset=tokenized_datasets["train"],  # Training dataset
        eval_dataset=tokenized_datasets["valid"],  # Evaluation dataset
        compute_metrics=compute_metrics,
        callbacks=[
            es_callback,
            gcs_callback,
        ],  # Stop if no improvement in 3 evaluations
    )

    # Train the model
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    prediction_outputs = trainer.predict(test_dataset=tokenized_datasets["test"])  # type: ignore
    best_checkpoint = trainer.state.best_model_checkpoint

    print("predictions")
    pprint(prediction_outputs.metrics)

    with open("best_checkpoint.json", "w") as f:
        json.dump(obj={"checkpoint": best_checkpoint}, fp=f)
    print(f"Best Checkpoint: {best_checkpoint} Saved.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="User Info CLI")
    parser.add_argument(
        "-e", "--epochs", type=int, default=3, help="number of train epochs"
    )
    parser.add_argument(
        "-tb", "--train_batch", type=int, default=8, help="number of train batches"
    )
    parser.add_argument(
        "-eb", "--eval_batch", type=int, default=64, help="number of eval batches"
    )
    parser.add_argument(
        "-gs", "--grad_steps", type=int, default=4, help="number of gradient accumulation steps"
    )

    parser.add_argument(
        "-ls", "--log_steps", type=int, default=10, help="number of log steps"
    )

    parser.add_argument(
        "-ss", "--save_steps", type=int, default=300, help="number of save steps"
    )
    parser.add_argument(
        "-es", "--eval_steps", type=int, default=300, help="number of eval steps"
    )

    parser.add_argument(
        "-r", "--resume", type=str, default=None, help="resume checkpoint uri"
    )
    parser.add_argument("-u", "--upload", action="store_true", help="Enable uploads")
    parser.add_argument("-m", "--model", type=str, choices=['bert', 'roberta'], default='bert')
    args = parser.parse_args()

    bkt_upload = False
    if args.upload:
        print(f"bucket upload enabled")
        bkt_upload = True
    else:
        print(f"bucket upload disabled")

    pprint(args)
    train(
        model=args.model,
        bkt_upload=bkt_upload,
        num_epochs=args.epochs,
        resume_checkpoint=args.resume,
        train_batch=args.train_batch,
        eval_batch=args.eval_batch,
        grad_steps=args.grad_steps,
        log_steps=args.log_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
    )
