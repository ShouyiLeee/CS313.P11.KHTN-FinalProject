# CS313.P11.KHTN-FinalProject

#-------------------------------------------------------

# Anomaly Detection for Quality Control System

This project focuses on building an Anomaly Detection system for Quality Control, leveraging state-of-the-art deep learning and machine learning techniques. The system is tested on three datasets:

1. **MvTecAD**: A comprehensive dataset for industrial anomaly detection.
2. **WFDD (Wooden Flaw Detection Dataset)**: A dataset designed for detecting flaws in wooden surfaces.
3. **GoodsAD**: A dataset for anomaly detection in common goods.

## Objective
The objective of this project is to classify images of objects or surfaces from the above datasets as either **normal** or **abnormal** using four distinct methods. The results are evaluated and compared to determine the most effective approach for anomaly detection.

## Methods
This project implements the following methods for anomaly detection:

1. **AutoEncoder**:
   - An unsupervised learning approach where the model learns to reconstruct normal samples.
   - Anomalies are identified based on reconstruction errors.

2. **KNN + ResNet**:
   - Combines the power of K-Nearest Neighbors (KNN) and ResNet.
   - ResNet is used for feature extraction, and KNN is applied for anomaly detection based on extracted features.

3. **ResNet + AutoEncoder**:
   - A hybrid method that combines the strengths of ResNet for feature extraction and AutoEncoder for anomaly reconstruction.

4. **SimpleNet**:
   - A lightweight neural network specifically designed for efficient anomaly detection.
   - SimpleNet is trained to distinguish between normal and abnormal samples directly.

## Datasets
### 1. MvTecAD
- **Description**: A dataset comprising high-resolution images of industrial objects and textures.
- **Anomalies**: Includes scratches, dents, and other manufacturing defects.

### 2. WFDD (Wooden Flaw Detection Dataset)
- **Description**: Images of wooden surfaces with and without flaws.
- **Anomalies**: Cracks, knots, and other imperfections.

### 3. GoodsAD
- **Description**: A dataset for anomaly detection in everyday goods.
- **Anomalies**: Includes defects such as missing parts, scratches, or incorrect assembly.

## Workflow
1. **Data Preprocessing**:
   - Images from the datasets are preprocessed to ensure compatibility with the models.
   - Augmentation techniques are applied to enhance model robustness.

2. **Model Training**:
   - Each method is trained separately on the datasets.
   - Training involves hyperparameter tuning to optimize performance.

3. **Evaluation**:
   - Models are evaluated using metrics such as Accuracy, Precision, Recall, and F1-score.
   - Reconstruction errors (for AutoEncoder-based methods) and feature similarity (for KNN-based methods) are analyzed.

4. **Comparison**:
   - The results of all methods are compared to identify the most effective approach for anomaly detection.

## Results
The performance of each method on the three datasets is summarized, highlighting strengths and weaknesses. Detailed results and insights are provided in the `Results` section of the project repository.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/anomaly-detection-qcs.git
   ```
2. Navigate to the project directory:
   ```bash
   cd anomaly-detection-qcs
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the training script:
   ```bash
   python train.py --method <method_name> --dataset <dataset_name>
   ```
   Replace `<method_name>` with `autoencoder`, `knn_resnet`, `resnet_autoencoder`, or `simplenet`, and `<dataset_name>` with `mvtec`, `wfdd`, or `goodsad`.

5. View results:
   ```bash
   python evaluate.py --method <method_name> --dataset <dataset_name>
   ```

## Repository Structure
- `datasets/`: Contains the datasets used in this project.
- `models/`: Includes implementations of the four methods.
- `scripts/`: Training and evaluation scripts.
- `results/`: Stores evaluation results and visualizations.
- `README.md`: Project description.

## Future Work
- Extend the system to handle other types of anomalies and datasets.
- Improve the efficiency of the models for real-time anomaly detection.
- Explore semi-supervised and self-supervised approaches for anomaly detection.

## Contributors
- [Your Name](https://github.com/your-profile)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
