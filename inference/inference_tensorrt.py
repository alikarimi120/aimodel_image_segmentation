import json
import os
import random
import shutil

import cv2
import yaml
from ultralytics import YOLO


class ConfigLoader:
    def __init__(self, config_path):
        # Load configuration from the provided JSON file
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)

    def get_config(self):
        # Return the loaded configuration
        return self.config


class YOLOInference:
    def __init__(self, config_path):
        self.config = config_path
        self.root_path = self.config['root_output_path']
        self.font_size = self.config['font_size']
        self.model_path = os.path.join(self.root_path, "weights/best.engine")
        self.predict_path = os.path.join(self.root_path, 'predict')

        self.data_config = self.load_data_config()
        self.labels = {int(key): value for key, value in self.data_config['names'].items()}
        self.model = YOLO(self.model_path, task='detect')
        self.colors = self.generate_colors(len(self.labels))
        self.create_output_directory()

    def load_data_config(self):
        with open(self.config['data_path'], 'r') as file:
            return yaml.safe_load(file)

    def create_output_directory(self):
        os.makedirs(self.predict_path, exist_ok=True)

    @staticmethod
    def generate_colors(num_classes):
        random.seed(0)
        return {i: [random.randint(0, 255) for _ in range(3)] for i in range(num_classes)}

    def draw_boxes(self, image, boxes_items, confidences_items):
        for i, box_item in enumerate(boxes_items):
            x1, y1, x2, y2, class_id = box_item
            color = self.colors[class_id]
            label = f'{self.labels[class_id]} {confidences_items[i]:.2f}'
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, label, (x1, y1 - 10), font, self.font_size, color, 2, cv2.LINE_AA)
        return image

    def infer_image(self, input_image_path):
        image = cv2.imread(input_image_path)
        image_height, image_width = image.shape[:2]
        results = self.model(input_image_path, imgsz=640, conf=0.25)  # Ensure correct inference settings

        boxes_items, confidences_items = self.extract_boxes_and_confidences(results)
        image_with_boxes = self.draw_boxes(image.copy(), boxes_items, confidences_items)
        return image_with_boxes, boxes_items, confidences_items, image_width, image_height

    @staticmethod
    def extract_boxes_and_confidences(results):
        boxes_items = []
        confidences_items = []
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].cpu().numpy())
            class_id = int(result.cls[0].cpu().numpy())
            confidence_item = float(result.conf[0].cpu().numpy())
            boxes_items.append([x1, y1, x2, y2, class_id])
            confidences_items.append(confidence_item)
        return boxes_items, confidences_items

    def save_image(self, image, input_mage_path):
        base_name = os.path.basename(input_mage_path)
        original_save_path = os.path.join(self.predict_path, base_name)
        predicted_save_path = os.path.join(self.predict_path, f'predict_{base_name}')
        shutil.copy(input_mage_path, original_save_path)
        cv2.imwrite(predicted_save_path, image)

    def save_boxes(self, input_boxes, input_confidences, input_image_path, image_width, image_height):
        base_name = os.path.splitext(os.path.basename(input_image_path))[0]
        txt_path = os.path.join(self.predict_path, f'predict_{base_name}.txt')
        with open(txt_path, 'w') as f:
            for box_item, confidence_item in zip(input_boxes, input_confidences):
                x1, y1, x2, y2, class_id = box_item
                x_center, y_center, width, height = self.convert_to_normalized_format(x1, y1, x2, y2, image_width,
                                                                                      image_height)
                f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')

    @staticmethod
    def convert_to_normalized_format(x1, y1, x2, y2, image_width, image_height):
        x_center = ((x1 + x2) / 2) / image_width
        y_center = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        return x_center, y_center, width, height


# Usage Example
if __name__ == "__main__":
    config_loader = ConfigLoader('../config/config.json')
    config = config_loader.get_config()
    inference = YOLOInference(config)

    image_path = '../1.jpg'
    output_image, boxes, confidences, img_width, img_height = inference.infer_image(image_path)

    # Save the output image
    inference.save_image(output_image, image_path)

    # Save the bounding box details
    inference.save_boxes(boxes, confidences, image_path, img_width, img_height)

    # Print the box details
    for box, confidence in zip(boxes, confidences):
        print(f'Box: {box}, Confidence: {confidence}')
