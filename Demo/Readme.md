# Anomaly Detection in Quality Control System - Web Demo

This web demo visualizes the results of our Anomaly Detection system for Quality Control. It provides an interactive interface for users to upload images and see whether they are classified as **normal** or **abnormal**.

## Features
- Upload images of objects or surfaces.
- Display predictions from four different methods:
  - **AutoEncoder**
  - **ResNet + AutoEncoder**
- Visualize reconstructed images (for AutoEncoder-based methods).
- Show heatmaps highlighting anomalous regions.

## How It Works
1. **Upload Image**:
   - Users can upload an image of an object or surface to be analyzed.

2. **Processing**:
   - You can choose the dataset and object type to detect. 
   - The system processes the image through the selected models.

4. **Results Display**:
   - The system outputs:
     - Prediction: Normal or Abnormal.
     - Confidence score.
     - Heatmap of detected anomalies.
     - Mask

## Screenshots
### Main Interface
![Main Interface](https://github.com/ShouyiLeee/CS313.P11.KHTN-FinalProject/blob/main/assets/Home.png)

### Results Page
![Results Page](https://github.com/ShouyiLeee/CS313.P11.KHTN-FinalProject/blob/main/assets/Predict.png)

## Usage

1. **Install Dependencies**:
   Follow the instructions in the `requirements.md` file to set up the environment.

2. **Run the Web App**:
   ```bash
   streamlit run DemoWeb.py
   ```

3. **Access the Demo**:
   Open a browser and navigate to:
   ```
   http://localhost:8501
   ```

4. **Upload and Analyze Images**:
   Use the interface to upload images and view results.


