# CS313.P11.KHTN-FinalProject


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
- **link** : [MvTecAD](https://www.kaggle.com/datasets/ipythonx/mvtec-ad)


### 2. WFDD (Wooden Flaw Detection Dataset)
- **Description**: Images of wooden surfaces with and without flaws.
- **Anomalies**: Cracks, knots, and other imperfections.
-  **link** : [WFDD](https://github.com/cqylunlun/GLASS)

### 3. GoodsAD
- **Description**: A dataset for anomaly detection in everyday goods.
- **Anomalies**: Includes defects such as missing parts, scratches, or incorrect assembly.
-  **link** : [GoodsAD](https://github.com/jianzhang96/GoodsAD)

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



## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/ShouyiLeee/CS313.P11.KHTN-FinalProject.git
   ```
2. Navigate to the project directory:
   ```bash
   CS313.P11.KHTN-FinalProject
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Repository Structure
- `datasets/`: Contains the datasets used in this project.
- `models/`: Includes implementations of the four methods.
- `scripts/`: Training and evaluation scripts.
- `results/`: Stores evaluation results and visualizations.
- `README.md`: Project description.

## Results
The performance of each method on the three datasets is summarized, highlighting strengths and weaknesses. Detailed results and insights are provided in the `Results` section of the project repository.

| Object         | AutoEncoder | KNN + ResNet | ResNet + AutoEncoder | SimpleNet |
|----------------|-------------|--------------|----------------------|-----------|
| cigarette_box  | 0.69        | 0.69         | 0.86                 | 0.81      |
| drink_bottle   | 0.51        | 0.51         | 0.58                 | 0.61      |
| drink_can      | 0.56        | 0.56         | 0.68                 | 0.60      |
| food_bottle    | 0.60        | 0.63         | 0.72                 | 0.73      |
| food_box       | 0.59        | 0.57         | 0.66                 | 0.71      |
| food_package   | 0.56        | 0.50         | 0.58                 | 0.56      |


## Future Work
- Extend the system to handle other types of anomalies and datasets.
- Improve the efficiency of the models for real-time anomaly detection.
- Explore semi-supervised and self-supervised approaches for anomaly detection.

## Contributors
- [ShouyiLeee](https://github.com/ShouyiLeee)

