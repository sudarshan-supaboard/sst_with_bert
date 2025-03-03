import torch
import gc
import shutil
from transformers import TrainerCallback
from google.cloud import storage
from config import Config

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
            val_losses = [log["eval_loss"] for log in state.log_history if "eval_loss" in log]
            if val_losses:
                current_loss = val_losses[-1]  # Get the most recent training loss
                if current_loss < self.best_loss - self.min_delta:
                    self.best_loss = current_loss
                    self.counter = 0  # Reset patience counter if loss improves
                else:
                    self.counter += 1  # Increment counter if no improvement

                if self.counter >= self.patience:
                    control.should_training_stop = True
                    print("\nEarly stopping triggered due to no improvement in training loss!")



class GCSUploadCallback(TrainerCallback):
    def __init__(self, bucket_name=Config.BUCKET_NAME, checkpoint_dir=Config.OUTPUT_DIR):
        self.bucket_name = bucket_name
        self.checkpoint_dir = checkpoint_dir  # Local directory where checkpoints are saved
        self.storage_client = storage.Client(project=Config.PROJECT_ID)
        self.bucket = self.storage_client.bucket(bucket_name)

    def upload_to_gcs(self, local_path):
        """Uploads a file or directory to GCS recursively."""
        zip_file = f"{local_path}.zip"
        shutil.make_archive(local_path, 'zip', local_path)
        print(f"Zipped {local_path} -> {zip_file}")
        
        blob = self.bucket.blob(zip_file)
        blob.upload_from_filename(zip_file)
        print(f"Uploaded {local_path} to gs://{self.bucket_name}/{zip_file}")
        
        return zip_file
        
    def on_save(self, args, state, control, **kwargs):
        """Triggered whenever a checkpoint is saved."""
        print("Saving checkpoint, uploading to GCS...")
        
        # Upload the entire checkpoint directory
        uploaded_file = self.upload_to_gcs(self.checkpoint_dir)

        print(f"Checkpoint uploaded to gs://{self.bucket_name}/{uploaded_file}")


