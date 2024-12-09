import streamlit as st
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Load mô hình đã huấn luyện
def load_model():
    # Tải mô hình Autoencoder hoặc ResNet50
    model = torch.load("model.pth", map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

# Chuyển đổi dữ liệu hình ảnh
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Dự đoán hình ảnh
def predict(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        # Tùy thuộc vào cách bạn triển khai mô hình, xử lý output tại đây
        anomaly_score = torch.mean((image_tensor - output) ** 2).item()
    return anomaly_score

# Hiển thị heatmap
def generate_heatmap(image_tensor, output_tensor):
    diff = (image_tensor - output_tensor).squeeze().numpy()
    plt.imshow(np.transpose(diff, (1, 2, 0)), cmap="hot")
    plt.colorbar()
    plt.title("Anomaly Heatmap")
    plt.axis("off")
    return plt

# Giao diện website
st.title("Anomaly Detection Demo")
st.sidebar.title("Menu")
page = st.sidebar.radio("Chọn chức năng", ["Upload & Kết quả", "Mô tả trực quan"])

if page == "Upload & Kết quả":
    st.header("Upload dữ liệu và nhận kết quả")
    uploaded_file = st.file_uploader("Tải lên hình ảnh", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, caption="Ảnh gốc", use_column_width=True)
        image_tensor = preprocess_image(uploaded_file)
        
        anomaly_score = predict(image_tensor)
        st.write(f"Anomaly Score: {anomaly_score:.4f}")
        if anomaly_score > 0.5:  # Ngưỡng xác định bất thường (tùy chỉnh)
            st.error("Kết quả: Bất thường!")
        else:
            st.success("Kết quả: Bình thường!")

if page == "Mô tả trực quan":
    st.header("Mô tả trực quan vùng bất thường")
    uploaded_file = st.file_uploader("Tải lên hình ảnh", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Ảnh gốc", use_column_width=True)
        image_tensor = preprocess_image(uploaded_file)
        
        with torch.no_grad():
            output_tensor = model(image_tensor)
        
        heatmap_plot = generate_heatmap(image_tensor, output_tensor)
        st.pyplot(heatmap_plot)
