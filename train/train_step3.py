import csv
import os
import argparse

import shutil
import json
from ultralytics import YOLO


class ConfigLoader:
    def __init__(self, config_path):
        # Load configuration from the provided JSON file
        with open(config_path, 'r') as configure_file:
            self.config = json.load(configure_file)

    def get_config(self):
        # Return the loaded configuration
        return self.config


class DataProcessor:
    def __init__(self, config_path):
        self.flops = None
        self.parameters = None
        config_loader = ConfigLoader(config_path)
        self.config = config_loader.get_config()
        self.root_path = self.config['root_output_path']
        self.data_path = self.config['data_path']
        self.results_file = self.root_path + "/results.csv"
        self.tensorboard_path = self.root_path + "/tensorboard"
        self.images_path = self.root_path + "/images"
        self.logs_path = self.root_path + "/logss"

        self.model_path = self.root_path + "/weights/best.pt"
        self.output_file = self.root_path + "/logss/id_training_metrics.csv"
        self.training_metrics_file = self.root_path + "/logss/temp.csv"
        self.classification_report_file = self.root_path + "/logss/id_classification_report.csv"
        self.classification_report2_file = self.root_path + "/logss/id_classification_report2.csv"
        self.confusion_matrix_file = self.root_path + "/logss/id_confusion_matrix.csv"

        self.epoch = 0
        self.class_name_list = []
        self.list_class_names = []
        self.list_class = []
        self.results_data = []
        self.training_metrics_data = []
        self.create_directories()

    def create_directories(self):

        os.makedirs(self.tensorboard_path, exist_ok=True)
        os.makedirs(self.images_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)

    def process_input_file(self, input_file_txt):
        with open(input_file_txt, 'r') as file:
            lines = file.readlines()

        with open(self.training_metrics_file, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            flag = 0
            count_lines = 0
            export_complete = 0
            epoch = 0
            all_class_head = 0
            all_class_value = 0

            for line in lines:
                count_lines += 1
                if line.strip()[0:11] == "Ultralytics":
                    export_complete = 1
                    flag = 10
                if export_complete == 0:
                    continue
                if line.strip()[0:21] == "Model summary (fused)":
                    flag = 0
                    all_class_value = 0
                    list_attr = []
                    epoch += 1
                    list_class = []
                if line.strip()[0:7] == "WARNING":
                   continue
                    

                if flag == 0 and line.strip()[0:5] != 'Speed' and line.strip()[0:5] == 'Model':
                    line_attr = line.strip().split(",")
                    self.parameters = line_attr[1].strip().split(" ")[0]
                    self.flops = line_attr[3].strip().split(" ")[0]

                elif flag == 0 and line.strip()[0:5] != 'Speed':
                    line=line.replace("-","_")
                    parts = line.split()
                    if epoch == 1 and all_class_head == 0:
                        self.class_name_list.extend([
                            "precision micro", "precision macro", "recall micro", "recall macro",
                            "map50 micro", "map50 macro", "map50_95 micro", "map50_95 macro"
                        ])
                    elif epoch == 1:
                        self.class_name_list.extend([
                            f"precision {parts[0]}", f"recall {parts[0]}", f"map50 {parts[0]}", f"map50_95 {parts[0]}"
                        ])
                        self.list_class_names.append(parts[0])
                    all_class_head = 1


                    line_attr = [float(part) if '.' in part else int(part) for part in parts if
                                 part.replace('.', '', 1).isdigit() or '-' in part.replace('.', '', 1)]
                    list_class.append(line_attr[1])
                    print(line_attr)
                    if all_class_value == 0:
                        list_attr.extend(["", line_attr[2], "", line_attr[3], "", line_attr[4], "", line_attr[5]])
                    else:
                        list_attr.extend([line_attr[2], line_attr[3], line_attr[4], line_attr[5]])
                    all_class_value = 1

                elif flag == 0 and line.strip()[0:5] == 'Speed':
                    if epoch == 1:
                        csvwriter.writerow(self.class_name_list)

                    precision_total = sum(
                        list_class[counter_class] * list_attr[z]
                        for counter_class, z in enumerate(range(8, len(list_attr), 4), start=1)
                    )
                    list_attr[0] = precision_total / list_class[0]

                    recall_total = sum(
                        list_class[counter_class] * list_attr[z]
                        for counter_class, z in enumerate(range(9, len(list_attr), 4), start=1)
                    )
                    list_attr[2] = recall_total / list_class[0]

                    map50_total = sum(
                        list_class[counter_class] * list_attr[z]
                        for counter_class, z in enumerate(range(10, len(list_attr), 4), start=1)
                    )
                    list_attr[4] = map50_total / list_class[0]

                    map5095_total = sum(
                        list_class[counter_class] * list_attr[z]
                        for counter_class, z in enumerate(range(11, len(list_attr), 4), start=1)
                    )
                    list_attr[6] = map5095_total / list_class[0]

                    csvwriter.writerow(list_attr)
                    flag += 1

            self.list_class = list_class  # Save list_class for later use

    def merge_files(self):
        self.results_data = self.read_csv(self.results_file)
        self.training_metrics_data = self.read_csv(self.training_metrics_file)

        merged_data = [
            {**self.results_data[i], **self.training_metrics_data[i]}
            for i in range(min(len(self.training_metrics_data), len(self.results_data)))
        ]

        if not merged_data:
            print("Error: No data merged")
            return

        with open(self.output_file, mode='w', newline='') as file:
            fieldnames = list(merged_data[0].keys())
            csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for row in merged_data:
                csv_writer.writerow(row)

    def create_classification_report(self):
        with open(self.classification_report_file, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["", "precision", "recall", "map50", "map50-95", "support"])

            for i, class_name in enumerate(self.list_class_names):
                if i + 1 < len(self.list_class):
                    csvwriter.writerow([
                        class_name,
                        self.training_metrics_data[-1][f'precision {class_name}'],
                        self.training_metrics_data[-1][f'recall {class_name}'],
                        self.training_metrics_data[-1][f'map50 {class_name}'],
                        self.training_metrics_data[-1][f'map50_95 {class_name}'],
                        self.list_class[i + 1]
                    ])
                else:
                    print(f"Skipping {class_name} due to insufficient list_class length")

            csvwriter.writerow([
                "macro avg",
                self.training_metrics_data[-1]['precision macro'],
                self.training_metrics_data[-1]['recall macro'],
                self.training_metrics_data[-1]['map50 macro'],
                self.training_metrics_data[-1]['map50_95 macro'],
                self.list_class[0]
            ])
            csvwriter.writerow([
                "weighted avg",
                self.training_metrics_data[-1]['precision micro'],
                self.training_metrics_data[-1]['recall micro'],
                self.training_metrics_data[-1]['map50 micro'],
                self.training_metrics_data[-1]['map50_95 micro'],
                self.list_class[0]
            ])

    def create_additional_report(self):
        with open(self.classification_report2_file, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["metric", "value"])
            csvwriter.writerow(["FLOPs", self.flops])
            csvwriter.writerow(["parameters", self.parameters])
            csvwriter.writerow(["train/box_loss", self.results_data[-1]["train/box_loss"]])
            csvwriter.writerow(["train/cls_loss", self.results_data[-1]["train/cls_loss"]])
            csvwriter.writerow(["train/dfl_loss", self.results_data[-1]["train/dfl_loss"]])
            csvwriter.writerow(["val/box_loss", self.results_data[-1]["val/box_loss"]])
            csvwriter.writerow(["val/cls_loss", self.results_data[-1]["val/cls_loss"]])
            csvwriter.writerow(["val/dfl_loss", self.results_data[-1]["val/dfl_loss"]])

    def create_confusion_matrix(self):
        model = YOLO(self.model_path)
        results = model.val(data=self.data_path)
        with open(self.confusion_matrix_file, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([""] + self.list_class_names + ["background"])

            for i, class_name in enumerate(self.list_class_names):
                list_numbers = [int(results.confusion_matrix.matrix[i][j]) for j in
                                range(results.confusion_matrix.matrix.shape[1])]
                csvwriter.writerow([class_name] + list_numbers)

            # Assuming the last row corresponds to the background class
            list_numbers = [int(results.confusion_matrix.matrix[len(self.list_class_names)][j]) for j in
                            range(results.confusion_matrix.matrix.shape[1])]
            csvwriter.writerow(["background"] + list_numbers)

    @staticmethod
    def read_csv(filepath, columns=None):
        data = []
        with open(filepath, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                stripped_row = {key.strip(): value.strip() for key, value in row.items()}
                if columns:
                    try:
                        selected_data = {col: stripped_row[col] for col in columns}
                        data.append(selected_data)
                    except KeyError as e:
                        print(f"Missing column in data: {e}")
                else:
                    data.append(stripped_row)
        return data

    def delete_files_and_directories(self):
        files_to_delete = [
            self.root_path + "/args.yaml",
            self.root_path + "/temp.csv",
            self.root_path + "/merged_results.csv",
            self.root_path + "/results.csv",
            self.root_path + "/logss/temp.csv",
            "output.txt"
        ]
        for file in files_to_delete:
            if os.path.exists(file):
                os.remove(file)

        for file in os.listdir(self.root_path):
            if file.endswith('.png') or file.endswith('.jpg') or file.startswith('events.'):
                os.remove(os.path.join(self.root_path, file))
        shutil.rmtree("runs")

    def run(self, input_file_txt):
        self.process_input_file(input_file_txt)
        self.merge_files()
        self.create_classification_report()
        self.create_additional_report()
        self.create_confusion_matrix()
        self.delete_files_and_directories()


if __name__ == "__main__":
    # Load configuration and setup data loaders and model trainer
    parser = argparse.ArgumentParser("automate_model")
    parser.add_argument("config", help="config file path.", type=str , default='../config/config.json')
    args = parser.parse_args()
    input_file = 'output.txt'
    processor = DataProcessor(args.config)
    processor.run(input_file)
