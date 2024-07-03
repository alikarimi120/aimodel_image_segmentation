import glob
import argparse

import json
from ultralytics import YOLO


class ConfigLoader:
    def __init__(self, config_path):
        # Load configuration from the provided JSON file
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)

    def get_config(self):
        # Return the loaded configuration
        return self.config


class YOLOTrainer:
    def __init__(self, config_path):
        # Initialize the YOLOTrainer with the path to the configuration file
        config_loader = ConfigLoader(config_path)
        self.config = config_loader.get_config()
        self.data_path = self.config["data_path"]
        self.root_path = self.config["root_output_path"]
        self.model_path = self.config["model_config"]["model_name"] # Set self.model_path
        self.weight_path = self.root_path + "/weights"
        self.model = self.load_model()  # Load the YOLO model

    def load_model(self):
        # Load the YOLO model using the path specified in the configuration file
        model = YOLO(self.model_path)
        return model

    def validate_model(self):
        # Validate the model using all weights in the specified weight path
        for i_weight_path in sorted(glob.glob(self.weight_path + "/ep*.pt")):
            model = YOLO(i_weight_path)
            model.val(data=self.data_path)


if __name__ == "__main__":
    # Create an instance of YOLOTrainer with the path to the config file
    parser = argparse.ArgumentParser("automate_model")
    parser.add_argument("config", help="config file path.", type=str , default='../config/config.json')
    args = parser.parse_args()
    trainer = YOLOTrainer(args.config)
    trainer.validate_model()
