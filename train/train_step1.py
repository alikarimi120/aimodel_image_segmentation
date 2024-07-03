import glob
import json
import os
import shutil
import sys
from io import StringIO
import argparse

import pandas as pd
import torch
from ptflops import get_model_complexity_info
from torchsummary import summary
from ultralytics import YOLO


class ConfigLoader:
    def __init__(self, config_path):
        # Load configuration from the provided JSON file
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)

    def get_config(self):
        # Return the loaded configuration
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config["device"]
        return self.config


class YOLOTrainer:
    def __init__(self, config_path):
        # Initialize the YOLOTrainer with the path to the configuration file
        config_loader = ConfigLoader(config_path)
        self.config = config_loader.get_config()
        self.root_path = self.config["root_output_path"]
        self.model_path = self.config["model_config"]["model_name"]
        self.data_path = self.config["data_path"]
        self.num_epochs = self.config["num_epochs"]
        self.batch_size = self.config["batch_size"]
        self.device = self.config["device"]
        self.img_size = self.config["image_size"]
        self.prepare_results_directory()  # Prepare the results directory
        self.model = self.load_model()  # Load the YOLO model
        self.model_directory_path = "runs/detect/aimodel"
        self.weight_path = "runs/detect/aimodel/weights"
        self.tensorboard_path = "runs/detect/aimodel/tensorboard"
        self.best_weight_path = "runs/detect/aimodel/weights/best.pt"
        self.model_path = ""  # Added title

    def load_model(self):
        # Load the YOLO model using the path specified in the configuration file
        model = YOLO(self.model_path)
        return model

    def prepare_results_directory(self):
        # Remove the existing results directory if it exists
        if os.path.exists(self.root_path):
            shutil.rmtree(self.root_path)

    def train_model(self):
        # Set the training parameters from the configuration file

        # Train the model with the specified parameters
        results = self.model.train(
            data=self.data_path,
            epochs=self.num_epochs,
            batch=self.batch_size,
            device=self.device,
            imgsz=self.img_size,
            save_period=1,
            exist_ok=True,
            name="aimodel"
        )

        return results

    def move_results_directory(self):
        # Move the results directory to the final destination
        shutil.move(self.model_directory_path, self.root_path)

    def change_last_epoch(self):
        # Validate the model using all weights in the specified weight path
        shutil.move(self.weight_path + "/last.pt", self.weight_path + "/epoch" + str(self.num_epochs) + ".pt")

    def export_onnx_model(self):
        # Export the trained model to ONNX format
        self.model.export(format="onnx")

    def export_tensorrt_model(self):
        # Export the trained model to TensorRT format
        self.model.export(format="tensorrt")

    def copy_event_files(self):
        # Copy all files starting with 'event' to the 'tensorboard' directory
        event_files = glob.glob(self.model_directory_path + '/event*')
        os.makedirs(self.tensorboard_path, exist_ok=True)
        for file in event_files:
            shutil.copy(file, self.tensorboard_path)

    def plot_model_summary(self):
        # Print the summary of the model
        torch.save(self.model.model.model, 'tensor.pt')
        self.model = torch.load('tensor.pt')
        buffer = StringIO()
        sys.stdout = buffer
        if isinstance(self.image_size, list) and len(self.image_size) == 2:
            self.image_size = tuple(self.image_size)
        elif isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)

        # Ensure the model and input tensor are on the same device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Generate model summary
        summary(self.model, input_size=(3, *self.image_size), device=device.type)

        sys.stdout = sys.__stdout__
        model_summary = buffer.getvalue()
        flops, params = get_model_complexity_info(self.model.model, (3, *self.image_size), as_strings=False,
                                                  print_per_layer_stat=False, verbose=False)

        with open(os.path.join(self.model_directory_path, "model_summary.txt"), "w") as f:
            f.write(f'Model Summary:\n{model_summary}\n')
            f.write(f'FLOPs: {flops}\n')
            f.write(f'Parameters: {params}\n')

        summary_data = {
            'metric': ['FLOPs', 'parameters', 'loss'],
            'value': [flops, params, self.epoch_data["validation loss"][-1]]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.model_directory_path, "model_summary_report.csv"), index=False)

    def export_model(self):
        model = YOLO(self.weight_path + "/best.pt")
        model.export(format="onnx")
        model.export(format="tensorrt")


if __name__ == "__main__":
    # Load configuration and setup data loaders and model trainer
    parser = argparse.ArgumentParser("automate_model")
    parser.add_argument("config", help="config file path.", type=str , default='../config/config.json')
    args = parser.parse_args()
    trainer = YOLOTrainer(args.config)
    # Train the model
    trainer.train_model()
    # Validate the model
    trainer.change_last_epoch()
    # Export the model to ONNX format
    # Plot model summary
    # trainer.plot_model_summary()
    # trainer.export_onnx_model()
    # Export the model to TensorRT format
    # trainer.export_tensorrt_model()
    # Copy event files to tensorboard directory
    trainer.copy_event_files()
    # Export Moodels
    trainer.export_model()
    # Move the results directory to the final path
    trainer.move_results_directory()
