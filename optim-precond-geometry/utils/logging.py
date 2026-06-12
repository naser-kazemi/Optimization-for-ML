import os
import csv

class CSVLogger:
    """Appends metric dicts to a CSV file, writing the header on first use."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.file_exists = os.path.exists(filepath)
        self.fieldnames = None

    def log(self, metrics_dict):
        if not self.fieldnames:
            self.fieldnames = list(metrics_dict.keys())

            dir_name = os.path.dirname(self.filepath)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            if not self.file_exists:
                with open(self.filepath, mode='w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()
                self.file_exists = True

        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics_dict)


class WandbLogger:
    """Logs to Weights & Biases when enabled, falling back to no-op if it's missing."""
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
