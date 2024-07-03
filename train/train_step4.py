import os
import random
import argparse

import cv2
import json
import yaml
from ultralytics import YOLO


class ConfigLoader:
    def __init__(self, config_path):
        # Load configuration from the provided JSON file
        with open(config_path, 'r') as configure_file:
            self.config = json.load(configure_file)

    def get_config(self):
        # Return the loaded configuration
        return self.config


class YOLOVisualizer:
    def __init__(self, config_path):
        config_loader = ConfigLoader(config_path)
        self.config = config_loader.get_config()

        self.root_path = self.config['root_output_path']
        self.data_config_path = self.config['data_path']
        self.images_path = self.root_path + "/images"
        self.model_path = self.root_path + "/weights/best.pt"
        self.train_limit = int(self.config['train_limit'])
        self.test_limit = int(self.config['test_limit'])
        self.font_size = float(self.config['font_size'])  # Read the font size from the config file

        os.makedirs(self.images_path, exist_ok=True)
        self.train_images_path, self.val_images_path, self.test_images_path, self.labels = self.read_data_paths_and_labels()
        self.colors = self.generate_colors(len(self.labels))
        self.model = YOLO(self.model_path)
        self.train_images, self.train_actual_masks = self.load_images_and_masks(self.train_images_path)
        self.test_images, self.test_actual_masks = self.load_images_and_masks(self.test_images_path)
        self.train_images, self.train_actual_masks = self.limit_images_and_masks(self.train_images,
                                                                                 self.train_actual_masks,
                                                                                 self.train_limit)
        self.test_images, self.test_actual_masks = self.limit_images_and_masks(self.test_images, self.test_actual_masks,
                                                                               self.test_limit)

    def read_data_paths_and_labels(self):
        with open(self.data_config_path, 'r') as file:
            data_config = yaml.safe_load(file)
        train_images_path = os.path.join(self.config['data_path'].replace("data.yaml", ""), data_config['train'])
        val_images_path = os.path.join(self.config['data_path'].replace("data.yaml", ""), data_config['val'])
        test_images_path = os.path.join(self.config['data_path'].replace("data.yaml", ""),
                                        data_config['test']) if 'test' in data_config else None
        labels = {int(key): value for key, value in data_config['names'].items()}
        return train_images_path, val_images_path, test_images_path, labels

    def load_images_and_masks(self, images_path):
        image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg') or f.endswith('.png')]
        valid_images = []
        valid_masks = []
        labels_path = images_path.replace("images", "labels")
        for img in image_files:
            img_path = os.path.join(images_path, img)
            mask_file = os.path.splitext(img)[0] + '.txt'
            mask_path = os.path.join(labels_path, mask_file)
            if os.path.exists(mask_path):
                valid_images.append(img_path)
                valid_masks.append(self.read_masks(mask_path))
        return valid_images, valid_masks

    @staticmethod
    def read_masks(mask_path):
        masks = []
        with open(mask_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_id = int(parts[0])
                polygon_points = list(map(float, parts[1:]))
                masks.append((class_id, polygon_points))
        return masks

    @staticmethod
    def limit_images_and_masks(images, masks, limit):
        if len(images) > limit:
            indices = random.sample(range(len(images)), limit)
            images = [images[i] for i in indices]
            masks = [masks[i] for i in indices]
        return images, masks

    @staticmethod
    def generate_colors(num_classes):
        random.seed(0)
        return {i: [random.randint(0, 255) for _ in range(3)] for i in range(num_classes)}

    def draw_masks(self, image, masks, labels):
        try:
            overlay = image.copy()
            for mask in masks:
                class_id, polygon_points = mask
                color = self.colors[class_id]
                points = [(int(polygon_points[i]), int(polygon_points[i + 1])) for i in range(0, len(polygon_points), 2)]
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(overlay, [points], color)
                cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
            return image
        except Exception as e:
            print(e)

    def plot_prediction_vs_actual(self, image_path, actual_masks, predicted_masks, labels, prefix):
        try:
            image = cv2.imread(image_path)
            image_actual = self.draw_masks(image.copy(), actual_masks, labels)  # Draw actual masks
            image_predicted = self.draw_masks(image.copy(), predicted_masks, labels)  # Draw predicted masks

            # Combine actual and predicted images side by side
            combined_image = cv2.hconcat([image_actual, image_predicted])

            # Add labels above images
            label_image = cv2.copyMakeBorder(combined_image, 60, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(label_image, "Actual", (image_actual.shape[1] // 2 - 50, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_size, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(label_image, "Prediction", (image_actual.shape[1] + image_predicted.shape[1] // 2 - 100, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (0, 0, 0), 2, cv2.LINE_AA)

            # Save the combined image with the appropriate prefix in the filename
            save_path = os.path.join(self.images_path, f'{prefix}_{os.path.basename(image_path)}')
            cv2.imwrite(save_path, label_image)
        except Exception as e:
            print(e)

    def predict_masks(self, images):
        all_predictions = []
        for image_path in images:
            results = self.model(image_path, imgsz=640, conf=0.25, task='segment')  # Ensure correct inference settings
            predictions = []
            for result in results[0].masks:
                class_id = int(result.cls[0].cpu().numpy())
                polygon_points = result.xy.cpu().numpy().flatten().tolist()
                predictions.append((class_id, polygon_points))
            all_predictions.append(predictions)
        return all_predictions

    def visualize_predictions(self):
        train_predicted_masks = self.predict_masks(self.train_images)
        for image_path, actual, predicted in zip(self.train_images, self.train_actual_masks,
                                                 train_predicted_masks):
            self.plot_prediction_vs_actual(image_path, actual, predicted, self.labels, 'train')

        if self.test_images_path:
            test_predicted_masks = self.predict_masks(self.test_images)
            for image_path, actual, predicted in zip(self.test_images, self.test_actual_masks,
                                                     test_predicted_masks):
                self.plot_prediction_vs_actual(image_path, actual, predicted, self.labels, 'test')


if __name__ == "__main__":
    # Load configuration and setup data loaders and model trainer
    parser = argparse.ArgumentParser("automate_model")
    parser.add_argument("config", help="config file path.", type=str, default='../config/config.json')
    args = parser.parse_args()
    visualizer = YOLOVisualizer(args.config)
    visualizer.visualize_predictions()

