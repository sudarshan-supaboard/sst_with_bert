import torch
import gc
import os

from transformers import TrainerCallback
from google.cloud import storage

def clear_cache():
  torch.cuda.empty_cache()  # Free unused memory
  gc.collect()  # Run Python garbage collector

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def get_checkpoints_dir():
    return "./checkpoints"

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



class GCSUploadCallback(TrainerCallback):
    def __init__(self, bucket_name, checkpoint_dir="checkpoints"):
        self.bucket_name = bucket_name
        self.checkpoint_dir = checkpoint_dir  # Local directory where checkpoints are saved
        self.storage_client = storage.Client(project="true-sprite-412217")  # Initialize Google Cloud Storage client
        self.bucket = self.storage_client.bucket(bucket_name)

    def upload_to_gcs(self, local_path, gcs_path):
        """Uploads a file or directory to GCS recursively."""
        if os.path.isdir(local_path):  # Upload directory
            for root, _, files in os.walk(local_path):
                for file in files:
                    local_file = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file, local_path)
                    blob_path = os.path.join(gcs_path, relative_path)
                    
                    blob = self.bucket.blob(blob_path)
                    blob.upload_from_filename(local_file)
                    print(f"Uploaded {local_file} to gs://{self.bucket_name}/{blob_path}")
                    
        else:  # Upload single file
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to gs://{self.bucket_name}/{gcs_path}")
            
    def on_save(self, args, state, control, **kwargs):
        """Triggered whenever a checkpoint is saved."""
        print("Saving checkpoint, uploading to GCS...")
        
        # Define the GCS destination path
        gcs_path = f"{self.checkpoint_dir}/{state.global_step}/"
        
        # Upload the entire checkpoint directory
        self.upload_to_gcs(self.checkpoint_dir, gcs_path)

        print(f"Checkpoint uploaded to gs://{self.bucket_name}/{gcs_path}")



if __name__ == '__main__':
    print(f'Device: {get_device()}')