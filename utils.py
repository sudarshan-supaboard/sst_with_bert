import torch
import gc

from transformers import TrainerCallback

def clear_cache():
  torch.cuda.empty_cache()  # Free unused memory
  gc.collect()  # Run Python garbage collector

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


class EarlyStoppingTrainingLossCallback(TrainerCallback):
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def on_save(self, args, state, control, **kwargs):
        """Called at the end of every step to monitor training loss."""
        if state.log_history:
            train_losses = [log["loss"] for log in state.log_history if "loss" in log]
            if train_losses:
                current_loss = train_losses[-1]  # Get the most recent training loss
                if current_loss < self.best_loss - self.min_delta:
                    self.best_loss = current_loss
                    self.counter = 0  # Reset patience counter if loss improves
                else:
                    self.counter += 1  # Increment counter if no improvement

                if self.counter >= self.patience:
                    control.should_training_stop = True
                    print("\nEarly stopping triggered due to no improvement in training loss!")

