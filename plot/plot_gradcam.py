import json
import os
from glob import glob

import cv2
import numpy as np
import pytorch_grad_cam
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


class ConfigLoader:
    def __init__(self, config_path):
        # Load configuration from the provided JSON file
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)

    def get_config(self):
        # Return the loaded configuration
        return self.config


class GradCAMProcessor:
    def __init__(self, config_path):
        # Load the configuration and model
        self.config = self.load_config(config_path)
        self.model = self.load_model(self.config["model_config"]["model_name"])
        # Get the target layers for Grad-CAM
        self.target_layers = self.get_target_layers()
        # List of Grad-CAM types to be used
        self.grad_cam_list = self.config["grad_cam_list"]

    @staticmethod
    def load_config(config_path):
        # Load configuration from JSON file
        with open(config_path, 'r') as config_file:
            return json.load(config_file)

    @staticmethod
    def load_model(model_name):
        # Load the pre-trained model from a file
        return torch.load(model_name)

    def get_target_layers(self):
        # Identify target layers (Conv2d layers) in the model for Grad-CAM
        target_layers = []
        for name, layer in self.model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                print(name, layer)
                target_layers.append(layer)
        # Return the last Conv2d layer twice for different Grad-CAM types
        return [[target_layers[-1]], [target_layers[-1]]]

    def list_images(self, directory, ignore_pattern="result"):
        # List all images in the directory that do not contain the ignore pattern
        path_pattern = os.path.join(directory, "*" + self.config["suffix"])
        image_files = [file for file in glob(path_pattern) if
                       ignore_pattern.lower() not in os.path.basename(file).lower()]
        return image_files

    def process_images(self):
        # Process each image using the specified Grad-CAM types and target layers
        for grad_cam_type in self.grad_cam_list:
            for target_layer in self.target_layers:
                # Get the Grad-CAM function
                model_func = getattr(pytorch_grad_cam, grad_cam_type)
                cam = model_func(model=self.model, target_layers=target_layer)

                targets = None  # No specific targets are defined

                for image_path in self.list_images(self.config["image_directory"]):
                    # Read and preprocess the image
                    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
                    rgb_img = cv2.resize(rgb_img, self.config["image_size"])
                    rgb_img = np.float32(rgb_img) / 255
                    input_tensor = preprocess_image(rgb_img, mean=self.config["mean"],
                                                    std=self.config["std"])

                    # Generate Grad-CAM and overlay on the image
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                    grayscale_cam = grayscale_cam[0, :]
                    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                    # Save the result image with a modified filename
                    cv2.imwrite(image_path.replace(self.config["suffix"], "_result_" + grad_cam_type + ".jpg"),
                                visualization)


if __name__ == "__main__":
    # Initialize the processor with the configuration file and process images

    processor = GradCAMProcessor('../config/config_gradcam.json')
    processor.process_images()