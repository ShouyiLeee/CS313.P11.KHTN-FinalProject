# Anomaly Detection in Quality Control System - Web Demo

This web demo visualizes the results of our Anomaly Detection system for Quality Control. It provides an interactive interface for users to upload images and see whether they are classified as **normal** or **abnormal**.

## Features
- Upload images of objects or surfaces.
- Display predictions from four different methods:
  - **AutoEncoder**
  - **KNN + ResNet**
  - **ResNet + AutoEncoder**
  - **SimpleNet**
- Visualize reconstructed images (for AutoEncoder-based methods).
- Show heatmaps highlighting anomalous regions.

## How It Works
1. **Upload Image**:
   - Users can upload an image of an object or surface to be analyzed.

2. **Processing**:
   - The system processes the image through one or more selected models.

3. **Results Display**:
   - The system outputs:
     - Prediction: Normal or Abnormal.
     - Confidence score.
     - Reconstructed image (for AutoEncoder).
     - Heatmap of detected anomalies.

## Screenshots
### Main Interface
![Main Interface](./screenshots/main_interface.png)

### Results Page
![Results Page](./screenshots/results_page.png)

## Usage
1. **Clone Repository**:
   ```bash
   git clone https://github.com/your-repo/anomaly-detection-web-demo.git
   cd anomaly-detection-web-demo
   ```

2. **Install Dependencies**:
   Follow the instructions in the `requirements.md` file to set up the environment.

3. **Run the Web App**:
   ```bash
   python app.py
   ```

4. **Access the Demo**:
   Open a browser and navigate to:
   ```
   http://localhost:5000
   ```

5. **Upload and Analyze Images**:
   Use the interface to upload images and view results.

## Future Improvements
- Add support for batch image uploads.
- Enhance the heatmap visualization with more interpretability.
- Include a detailed log of predictions and system performance.

## Contributors
- [Your Name](https://github.com/your-profile)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
