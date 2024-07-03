import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ConfigLoader:
    def __init__(self, config_path):
        # Load configuration from the provided JSON file
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)

    def get_config(self):
        # Return the loaded configuration
        return self.config


class ConfusionMatrixVisualizer:
    def __init__(self, directory_path, confusion_csv_file_path):
        self.csv_file_path = confusion_csv_file_path
        self.directory_path = directory_path

    def plot_confusion_matrix(self, output_confusion_image_path):
        # Read the confusion matrix from CSV
        cm_df = pd.read_csv(self.csv_file_path, index_col=0)

        # Normalize the confusion matrix by rows (true labels)
        cm_normalized = cm_df.div(cm_df.sum(axis=1), axis=0)

        # Create annotations combining counts and percentages
        annotations = cm_df.copy().astype(str)
        for i in range(cm_df.shape[0]):
            for j in range(cm_df.shape[1]):
                annotations.iloc[i, j] = f'{cm_df.iloc[i, j]} ({cm_normalized.iloc[i, j]:.2%})'

        # Set up the matplotlib figure
        plt.figure(figsize=(10, 7))
        sns.set(font_scale=1.2)

        # Draw the heatmap with the combined annotations
        ax = sns.heatmap(cm_normalized, annot=annotations, fmt='', cmap="YlGnBu", cbar=True, annot_kws={"size": 14})

        # Set axis labels
        ax.set_xlabel('Predicted Labels', fontsize=16)
        ax.set_ylabel('True Labels', fontsize=16)
        ax.set_title('Confusion Matrix with Counts and Percentages', fontsize=18)
        # Save the figure
        plt.savefig(output_confusion_image_path)
        print(f"Confusion matrix plot saved as {output_confusion_image_path}")


if __name__ == "__main__":
    # Usage of the class to read data and confusion matrix

    config_loader = ConfigLoader('../config/config.json')
    config = config_loader.get_config()
    root_directory_path = config['root_output_path']

    csv_file_path = root_directory_path + '/logss/id_confusion_matrix.csv'
    output_image_path = root_directory_path + '/images/id_confusion_matrix_plot.png'
    visualizer = ConfusionMatrixVisualizer(root_directory_path, csv_file_path)
    visualizer.plot_confusion_matrix(output_image_path)
