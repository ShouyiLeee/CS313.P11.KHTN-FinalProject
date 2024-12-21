import streamlit as st
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import cv2

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score
import seaborn as sns
from PIL import Image

from Model import FeatCAE, resnet_feature_extractor


device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load mô hình đã huấn luyện
def load_model(model_path):
    model = FeatCAE(in_channels=1536, latent_dim=100).to(device)
    model.load_state_dict(torch.load(model_path))
    backbone = resnet_feature_extractor().to(device)
    return model, backbone


def decision_function(segm_map):  
    mean_top_10_values = []
    for map in segm_map:
        # Flatten the tensor
        flattened_tensor = map.reshape(-1)
        # Sort the flattened tensor along the feature dimension (descending order)
        sorted_tensor, _ = torch.sort(flattened_tensor,descending=True)
        # Take the top 10 values along the feature dimension
        mean_top_10_value = sorted_tensor[:10].mean()
        mean_top_10_values.append(mean_top_10_value)
    return torch.stack(mean_top_10_values)



#-----------------------------------------------------------------------------------------------------



# model = load_model()

# Chuyển đổi dữ liệu hình ảnh
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).to(device).unsqueeze(0)


#---------------------------------------------------------------------------------------------

# Giao diện website
st.title("Anomaly Detection Demo")
st.sidebar.title("Menu")

object_label_mapping = {
"GoodsAD": ["cigarette_box", "drink_bottle", "drink_can", "food_bottle", "food_box", "food_package"],
"WFDD": ["grey_cloth", "grid_cloth", "pink_flower", "yellow_cloth"],
"MvtecAD": ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"],
}

# Dropdown để chọn bộ dataset
data_label = st.sidebar.selectbox(
    "Chọn bộ dataset",
    list(object_label_mapping.keys()),  # Lấy danh sách data_label từ keys của dictionary
)

# Cập nhật danh sách object_label tương ứng với data_label đã chọn
object_label = st.sidebar.selectbox(
    "Chọn loại vật thể cần phân loại",
    object_label_mapping[data_label],  # Lấy danh sách object_label tương ứng
)

# Slider để chọn ngưỡng threshold
threshold = st.sidebar.slider(
    "Chọn ngưỡng xác định bất thường", 
    min_value=0.0, 
    max_value=5.0, 
    value=0.5, 
    step=0.01
)

# Hiển thị thông tin đã chọn
# st.subheader(f"Bộ dataset: {data_label}")
# st.subheader(f"Loại vật thể: {object_label}")
# st.subheader(f"Ngưỡng xác định bất thường: {threshold}")

st.markdown(f"""
<style>
    .info-box {{
        background-color: #f0f8ff; /* Màu nền nhạt */
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }}
</style>
<div class="info-box">
    <strong>Bộ dataset:</strong> {data_label}
</div>
<div class="info-box">
    <strong>Loại vật thể:</strong> {object_label}
</div>
<div class="info-box">
    <strong>Ngưỡng xác định bất thường:</strong> {threshold}
</div>
""", unsafe_allow_html=True)

model_root_path = 'D:/IT/GITHUB/CS313.P11.KHTN-FinalProject/model'
model_path = f'{model_root_path}/{data_label}/AE-Resnet_{data_label}_{object_label}.pth'
model, backbone = load_model(model_path)

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.header("Upload dữ liệu và nhận kết quả")
uploaded_file = st.file_uploader("Tải lên hình ảnh", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Ảnh gốc", use_container_width=True)
    image_tensor = preprocess_image(uploaded_file)
    
    with torch.no_grad():
        features = backbone(image_tensor)
        recon = model(features)
        segm_map = ((features - recon)**2).mean(axis=(1))[:,3:-3,3:-3]
        anomaly_score = decision_function(segm_map=segm_map)
        anomaly_score = anomaly_score.cpu().numpy()
        anomaly_predict = (anomaly_score >= threshold).astype(int)
        
        st.write(f"Anomaly Score: {anomaly_score[0]:.4f}")
        if anomaly_predict == 0:  # Ngưỡng xác định bất thường (tùy chỉnh)
            st.success("Kết quả: Bình thường!")
        else:
            st.error("Kết quả: Bất thường!")


    st.header("Visualize")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original Image
    axs[0].imshow(image_tensor.cpu().squeeze().permute(1, 2, 0))
    axs[0].set_title(f'Original')
    axs[0].axis("off")
    
    # Heatmap
    heat_map = torch.nn.functional.interpolate(     # Upscale by bi-linaer interpolation to match the original input resolution
                segm_map.unsqueeze(0),
                size=(224, 224),
                mode='bilinear'
            )
    axs[1].imshow(heat_map.cpu().squeeze().numpy(), cmap='jet')
    axs[1].set_title(f'Anomaly score: {anomaly_score[0]:.4f}')
    axs[1].axis("off")
    
    # Mask
    axs[2].imshow((heat_map.cpu().squeeze().numpy() > threshold ), cmap='gray') 
    axs[2].set_title('Segmentation map')
    axs[2].axis("off")
    
    st.pyplot(fig)
    



