import itertools
import json
import matplotlib.pyplot as plt
import pandas as pd


class ConfigLoader:
    def __init__(self, config_path):
        # Load configuration from the provided JSON file
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)

    def get_config(self):
        # Return the loaded configuration
        return self.config


class TrainingMetricsPlotter:
    def __init__(self, directory_path, training_metrics_file_path):
        self.file_path = training_metrics_file_path
        self.data = pd.read_csv(training_metrics_file_path)
        self.epochs = self.data['epoch'].to_numpy()
        self.directory_path = directory_path

        # Extract all columns dynamically, excluding 'Epoch'
        self.columns = self.data.columns[1:]

    @staticmethod
    def plot_metric(x, y, label, color, x_label, y_label, title, filepath):
        """
        Plots a single metric.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, label=label, color=color)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(filepath)
        plt.close()

    def plot_metrics(self, merged_plot_file_path):
        """
        Plots all metrics and saves them individually and combined.
        """
        num_metrics = len(self.columns)
        plt.figure(figsize=(int(num_metrics * 2.5), 12))
        plt.suptitle('Image Classification Results', fontsize=16)

        num_rows = (num_metrics + 3) // 4  # Calculate the number of rows needed

        colors = itertools.cycle(
            ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'pink', 'lime'])

        # Plot combined metrics
        for idx, column in enumerate(self.columns, start=1):
            plt.subplot(num_rows, 4, idx)
            metric_data = self.data[column].to_numpy()
            color = next(colors)
            plt.plot(self.epochs, metric_data, label=column, color=color)
            plt.xlabel('Epoch')
            plt.ylabel(column)
            plt.title(f'{column} Over Epochs')
            plt.legend()
            plt.grid(True)
            # Save individual plot
            self.plot_metric(self.epochs, metric_data, column, color, 'Epoch', column, f'{column} Over Epochs',
                             f'{self.directory_path}/images/id_{column.lower().replace(" ", "_").replace("/","_")}_plot.png')

        # Adjust layout and save combined plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(wspace=0.6, hspace=0.6)  # Increase space between plots
        merged_plot_file_path = merged_plot_file_path
        plt.savefig(merged_plot_file_path)

        return merged_plot_file_path


if __name__ == "__main__":
    # Usage of the class to read data and plot metrics

    config_loader = ConfigLoader('../config/config.json')
    config = config_loader.get_config()
    root_directory_path = config['root_output_path']

    combined_csv_file_path = root_directory_path + '/logss/id_training_metrics.csv'
    combined_plot_file_path = root_directory_path + '/images/id_training_metrics_plots.png'
    visualizer = TrainingMetricsPlotter(root_directory_path, combined_csv_file_path)
    visualizer.plot_metrics(combined_plot_file_path)
    print(f"All plots saved. Combined plot path: {combined_plot_file_path}")
