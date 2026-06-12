import os
import csv

class CSVLogger:
    """
    A lightweight, robust CSV Logger for local metric tracking.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.file_exists = os.path.exists(filepath)
        self.fieldnames = None

    def log(self, metrics_dict):
        if not self.fieldnames:
            self.fieldnames = list(metrics_dict.keys())
            
            # Ensure folder exists
            dir_name = os.path.dirname(self.filepath)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
                
            # If file doesn't exist, write headers
            if not self.file_exists:
                with open(self.filepath, mode='w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()
                self.file_exists = True

        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics_dict)


class WandbLogger:
    """
    A wrapper for Weights & Biases that catches ImportError and disabled states.
    """
    def __init__(self, use_wandb=False, project="optml-optimizer-geometry", entity=None, config=None):
        self.enabled = use_wandb
        if self.enabled:
            try:
                import wandb
                wandb.init(
                    project=project,
                    entity=entity,
                    config=config
                )
                print(f"Weights & Biases initialized on project '{project}'.")
            except ImportError:
                print("Warning: 'wandb' package is not installed. Falling back to local logging.")
                self.enabled = False

    def log(self, metrics_dict):
        if self.enabled:
            try:
                import wandb
                wandb.log(metrics_dict)
            except Exception as e:
                print(f"Error logging to wandb: {e}")
