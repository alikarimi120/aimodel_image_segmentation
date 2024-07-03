
# AI Model Image Segmentation Automatic

This repository offers a comprehensive solution for image detection using AI models, supporting PyTorch, ONNX, and TensorRT formats.

## Table of Contents

- [Setup](#setup)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Visualization](#visualization)
- [Results](#results)
- [File Structure](#file-structure)
- [License](#license)

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/ai-model-image-detection.git
   cd ai-model-image-detection
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

- **config.json:** Main configuration file for training and inference settings.
- **config_gradcam.json:** Configuration specific to Grad-CAM visualizations.

## Training

To train the model, run the `train.py` script:

```bash
python train.py
```

Training results, including metrics and logs, will be saved in the `results` directory.

## Inference

You can perform inference using different model formats:

- **PyTorch:**
  ```bash
  python inference_torch.py
  ```

- **ONNX:**
  ```bash
  python inference_onnx.py
  ```

- **TensorRT:**
  ```bash
  python inference_tensorrt.py
  ```

## Visualization

Various scripts are provided for visualizing training and inference results:

- **Detection Report:**
  ```bash
  python plot_detection_report.py
  ```
  ![Detection Report](results/images/id_detection_report0.png)
  ![Detection Report 2](results/images/id_detection_report1.png)

- **Confusion Matrix:**
  ```bash
  python plot_cm.py
  ```
  ![Confusion Matrix](results/images/id_confusion_matrix_plot.png)

- **Grad-CAM:**
  ```bash
  python plot_gradcam.py
  ```

- **Training Metrics:**
  ```bash
  python plot_training_metrics.py
  ```
  ![Training Metrics](results/images/id_training_metrics_plots.png)

- **t-SNE Visualization:**
  ```bash
  python plot_tsne.py
  ```
  ![t-SNE Visualization - Test Data](results/images/id_tsne_test.png)
  ![t-SNE Visualization - Train Data](results/images/id_tsne_train.png)

## Results

The `results` directory contains all outputs from training and inference processes:

- **results/images:** Visualizations of training and inference results.
- **Logs:** CSV files with detailed metrics and logs.
- **Model Summary:** Text file with the model architecture summary.
- **Predictions:** Predictions made by the model on test results/images.
  ![Predictions - Train](results/images/train_predictions.png)
  ![Predictions - Validation](results/images/validation_predictions.png)
- **TensorBoard:** TensorBoard logs for detailed analysis.
- **Weights:** Saved model weights for different formats and epochs.

## File Structure

```plaintext
.
├── config_gradcam.json
├── config.json
├── demo_onnx.py
├── demo_torch.py
├── inference_onnx.py
├── inference_tensorrt.py
├── inference_torch.py
├── model_converter.py
├── plot_detection_report.py
├── plot_cm.py
├── plot_gradcam.py
├── plot_training_metrics.py
├── plot_tsne.py
├── README.md
├── requirements.txt
├── results
│   ├── results/images
│   │   ├── id_detection_report0.png
│   │   ├── id_detection_report1.png
│   │   ├── id_confusion_matrix_plot.png
│   │   ├── id_train_accuracy_plot.png
│   │   ├── id_train_loss_plot.png
│   │   ├── id_tsne_test.png
│   │   ├── id_tsne_train.png
│   │   ├── id_validation_accuracy_plot.png
│   │   ├── id_validation_loss_plot.png
│   │   ├── train_predictions.png
│   │   └── validation_predictions.png
│   ├── labels.txt
│   ├── logs
│   │   ├── id_detection_report2.csv
│   │   ├── id_detection_report.csv
│   │   ├── id_confusion_matrix.csv
│   │   └── id_training_metrics.csv
│   ├── model_summary.txt
│   ├── tensorboard
│   │   ├── events.out.tfevents.1718801496.2612267.0
│   ├── updated_config.json
│   └── weights
│       ├── best_model.onnx
│       ├── best_model.pt
│       ├── model_epoch_10.pth
│       ├── model_epoch_8.pth
│       └── model_epoch_9.pth
└── train.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any questions or issues, please open an issue on GitHub or contact the maintainers.
