import torch
import numpy as np
import torch.nn.functional as F
import evaluate

from transformers import Trainer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from pprint import pprint

from model import model, tokenized_datasets

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = model.module.device
        labels = inputs.pop("labels")
        labels = labels.to(device)

        # Forward pass
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits

        # Example: Custom loss (Focal Loss)
        loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss
    

# Load multiple metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"], # type: ignore
        "f1": f1.compute(predictions=predictions, references=labels, average='weighted')["f1"], # type: ignore
    }

# Create TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",          # Output directory
    num_train_epochs=1,              # Total number of training epochs
    per_device_train_batch_size=8,  # Batch size per device during training
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=64,   # Batch size for evaluation
    warmup_ratio=0.1,                # Number of warmup steps for learning rate scheduler
    learning_rate=5e-5,
    weight_decay=0.01,               # Strength of weight decay
    logging_dir="./logs",            # Directory for storing logs
    logging_steps=1,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=5,
    eval_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="wandb",
    bf16=True,
)

# Create Trainer instance
trainer = CustomTrainer(
    model=model,                         # The instantiated 🤗 Transformers model to be trained
    args=training_args,                  # Training arguments, defined above
    train_dataset=tokenized_datasets['train'],         # Training dataset
    eval_dataset=tokenized_datasets['valid'],           # Evaluation dataset
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Stop if no improvement in 5 evaluations
)

# Train the model
trainer.train()

prediction_outputs = trainer.predict(test_dataset=tokenized_datasets['test'])

print("predictions")
pprint(prediction_outputs)

