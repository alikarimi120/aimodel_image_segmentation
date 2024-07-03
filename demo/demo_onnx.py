import json
import os
import random
import shutil
import time

import cv2
import gradio as gr
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
    def __init__(self, config_file):
        self.model_path = os.path.join(config_file['root_output_path'], "weights/best.onnx")
        self.font_size = config_file['font_size']
        self.root_path = config_file['root_output_path']
        with open(config_file['data_path'], 'r') as file:
            data_config = yaml.safe_load(file)
        self.labels = {int(key): value for key, value in data_config['names'].items()}
        self.model = YOLO(self.model_path, task='detect')
        self.colors = self.generate_colors(len(self.labels))
        self.predict_path = os.path.join(self.root_path, 'predict')
        self.create_output_directory()

    def create_output_directory(self):
        os.makedirs(self.predict_path, exist_ok=True)

    def generate_colors(self, num_classes):
        random.seed(0)
        return {i: [random.randint(0, 255) for _ in range(3)] for i in range(num_classes)}

    def draw_boxes(self, image, boxes, confidences):
        for i, box in enumerate(boxes):
            x1, y1, x2, y2, class_id = box
            color = self.colors[class_id]
            label = f'{self.labels[class_id]} {confidences[i]:.2f}'
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, label, (x1, y1 - 10), font, self.font_size, color, 2, cv2.LINE_AA)
        return image

    def infer_image(self, image_path):
        start_preprocess = time.time()
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]
        end_preprocess = time.time()

        start_inference = time.time()
        results = self.model(image_path, imgsz=640, conf=0.25)
        end_inference = time.time()

        start_post_process = time.time()
        boxes = []
        confidences = []
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].cpu().numpy())
            class_id = int(result.cls[0].cpu().numpy())
            confidence = float(result.conf[0].cpu().numpy())
            boxes.append([x1, y1, x2, y2, class_id])
            confidences.append(confidence)
        end_post_process = time.time()

        preprocess_time = end_preprocess - start_preprocess
        inference_time = end_inference - start_inference
        post_process_time = end_post_process - start_post_process

        image_with_boxes = self.draw_boxes(image.copy(), boxes, confidences)
        return image_with_boxes, boxes, confidences, img_width, img_height, preprocess_time, inference_time, post_process_time

    def save_image(self, image, image_path):
        base_name = os.path.basename(image_path)
        original_save_path = os.path.join(self.predict_path, base_name)
        predicted_save_path = os.path.join(self.predict_path, f'predict_{base_name}')
        shutil.copy(image_path, original_save_path)
        cv2.imwrite(predicted_save_path, image)

    def save_boxes(self, boxes, confidences, image_path, img_width, img_height):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join(self.predict_path, f'predict_{base_name}.txt')
        with open(txt_path, 'w') as f:
            for box, confidence in zip(boxes, confidences):
                x1, y1, x2, y2, class_id = box
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')
        return txt_path


# Gradio Interface

config_loader = ConfigLoader('../config/config.json')
config = config_loader.get_config()
inference = YOLOInference(config)


def predict_image(image):
    image_path = "temp_image.png"
    cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    output_image, boxes, confidences, img_width, img_height, preprocess_time, inference_time, post_process_time = inference.infer_image(
        image_path)
    inference.save_image(output_image, image_path)
    txt_path = inference.save_boxes(boxes, confidences, image_path, img_width, img_height)

    # Print the box details
    details = []
    for box, confidence in zip(boxes, confidences):
        details.append(f'Box: {box}, Confidence: {confidence}')

    return (output_image[:, :, ::-1],
            "\n".join(details),
            f"{preprocess_time:.4f} seconds",
            f"{inference_time:.4f} seconds",
            f"{post_process_time:.4f} seconds",
            txt_path)


demo = gr.Interface(
    fn=predict_image,
    inputs=gr.components.Image(type="numpy", label="Upload Image"),
    outputs=[
        gr.components.Image(type="numpy", label="Detected Image"),
        gr.components.Textbox(label="Detection Details"),
        gr.components.Textbox(label="Preprocessing Time"),
        gr.components.Textbox(label="Inference Time"),
        gr.components.Textbox(label="Postprocessing Time"),
        gr.components.File(label="Download Bounding Box Details")
    ],
    title="YOLO Object Detection",
    description="Upload an image to get object detection results and processing times."
)

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', share=False)
