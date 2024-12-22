import streamlit as st
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms
import os
import cv2

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score
import seaborn as sns
from PIL import Image

from Model import FeatCAE, resnet_feature_extractor, Autoencoder


device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load mô hình đã huấn luyện
def load_ResnetAE(model_path):
    model = FeatCAE(in_channels=1536, latent_dim=100).to(device)
    model.load_state_dict(torch.load(model_path))
    backbone = resnet_feature_extractor().to(device)
    return model, backbone


def load_AE(model_path):
    model = model = Autoencoder().to(device)
    model.load_state_dict(torch.load(model_path))
    return model


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
threshold_AE = st.sidebar.slider(
    "Chọn ngưỡng xác định bất thường", 
    min_value=0.0, 
    max_value=0.2, 
    value=0.5, 
    step=0.001,
    key = 'AE'
)

threshold_ResnetAE = st.sidebar.slider(
    "Chọn ngưỡng xác định bất thường", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.001,
    key = 'ResnetAE'
)


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
    <strong>Ngưỡng xác định bất thường AE:</strong> {threshold_AE}
</div>
<div class="info-box">
    <strong>Ngưỡng xác định bất thường ResnetAE:</strong> {threshold_ResnetAE}
</div>
""", unsafe_allow_html=True)

model_root_path = 'D:/IT/GITHUB/CS313.P11.KHTN-FinalProject/model'
AE_path = f'{model_root_path}/{data_label}/AE_{data_label}_{object_label}.pth'
ResnetAe_path = f'{model_root_path}/{data_label}/AE-Resnet_{data_label}_{object_label}.pth'

ResnetAE, backboneRAE = load_ResnetAE(ResnetAe_path)
AE = load_AE(AE_path)

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.header("Upload dữ liệu và nhận kết quả")
uploaded_file = st.file_uploader("Tải lên hình ảnh", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Ảnh gốc", use_container_width=True)
    image_tensor = preprocess_image(uploaded_file)
    if object_label == "zipper" or object_label == "screw" or object_label == "grid":
        image_tensor = image_tensor.repeat(1, 3, 1, 1)
    
    with torch.no_grad():

        AE_recon = AE(image_tensor)
        AE_anomaly_score =((image_tensor - AE_recon)**2).mean(axis=(1))[:,0:-10,0:-10].mean()
        AE_anomaly_score = AE_anomaly_score.cpu().numpy()
        AE_anomaly_predict = (AE_anomaly_score >= threshold_AE).astype(int) 
        st.write(f"AE Anomaly Score: {AE_anomaly_score:.4f}")
        if AE_anomaly_predict == 0:  # Ngưỡng xác định bất thường (tùy chỉnh)
            st.success("Kết quả: Bình thường!")
        else:
            st.error("Kết quả: Bất thường!")
#-------------------------------------------------------------------------------------------
        
        features_RAE = backboneRAE(image_tensor)
        RAE_recon = ResnetAE(features_RAE)
        segm_map = ((features_RAE - RAE_recon)**2).mean(axis=(1))[:,3:-3,3:-3]
        RAE_anomaly_score = decision_function(segm_map=segm_map)
        RAE_anomaly_score = RAE_anomaly_score.cpu().numpy()
        RAE_anomaly_predict = (RAE_anomaly_score >= threshold_ResnetAE).astype(int)   
            
        st.write(f"ResnetAE Anomaly Score: {RAE_anomaly_score[0]:.4f}")
        if RAE_anomaly_predict == 0:  # Ngưỡng xác định bất thường (tùy chỉnh)
            st.success("Kết quả: Bình thường!")
        else:
            st.error("Kết quả: Bất thường!")


    st.header("Visualize")
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[0.5, 1, 1, 1], wspace=0.3, hspace=0.4)


    # Model Names
    # AE Title
    ax_title_ae = fig.add_subplot(gs[0, 0])
    ax_title_ae.text(0.5, 0.5, 'AutoEncoder', fontsize=20, fontweight='bold', ha='center', va='center')
    ax_title_ae.axis("off")

    # ResNet Title
    ax_title_resnet = fig.add_subplot(gs[1, 0])
    ax_title_resnet.text(0.5, 0.5, 'ResNetAE', fontsize=20, fontweight='bold', ha='center', va='center')
    ax_title_resnet.axis("off")

    # AE Results
    # Original Image
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(image_tensor.cpu().squeeze().permute(1, 2, 0).numpy())
    ax1.set_title('Original', fontsize=12)
    ax1.axis("off")

    # Heatmap
    ax2 = fig.add_subplot(gs[0, 2])
    heat_map_ae = ((image_tensor - AE_recon) ** 2).mean(axis=(1))
    im2 = ax2.imshow(heat_map_ae.cpu().squeeze().numpy(), cmap='jet')
    ax2.set_title('Heat Map', fontsize=12)
    ax2.axis("off")

    # Mask
    ax3 = fig.add_subplot(gs[0, 3])
    mask_ae = heat_map_ae.cpu().squeeze().numpy() > threshold_AE
    ax3.imshow(mask_ae, cmap='gray')
    ax3.set_title('Segmentation Map', fontsize=12)
    ax3.axis("off")

    # ResNet Results
    # Original Image
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(image_tensor.cpu().squeeze().permute(1, 2, 0).numpy())
    ax4.set_title('Original', fontsize=12)
    ax4.axis("off")

    # Heatmap
    ax5 = fig.add_subplot(gs[1, 2])
    heat_map_resnet = torch.nn.functional.interpolate(
        segm_map.unsqueeze(0),
        size=(224, 224),
        mode='bilinear'
    )
    im5 = ax5.imshow(heat_map_resnet.cpu().squeeze().numpy(), cmap='jet')
    ax5.set_title('Heat Map', fontsize=12)
    ax5.axis("off")

    # Mask
    ax6 = fig.add_subplot(gs[1, 3])
    mask_resnet = heat_map_resnet.cpu().squeeze().numpy() > threshold_ResnetAE
    ax6.imshow(mask_resnet, cmap='gray')
    ax6.set_title('Segmentation Map', fontsize=12)
    ax6.axis("off")

    # Adjust layout and show plot
    plt.tight_layout()
    st.pyplot(fig)
            
    # fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # # AE Model Results
    # # Original Image
    # axs[0, 0].imshow(image_tensor.cpu().squeeze().permute(1, 2, 0))
    # axs[0, 0].set_title(f'Original (AE)')
    # axs[0, 0].axis("off")

    # # Heatmap
    # heat_map_ae = ((image_tensor - AE_recon)**2).mean(axis=(1))
    # axs[0, 1].imshow(heat_map_ae.cpu().squeeze().numpy(), cmap='jet')
    # axs[0, 1].set_title(f'Heat Map (AE)')
    # axs[0, 1].axis("off")

    # # Mask
    # axs[0, 2].imshow((heat_map_ae.cpu().squeeze().numpy() > threshold_AE), cmap='gray')
    # axs[0, 2].set_title('Segmentation map (AE)')
    # axs[0, 2].axis("off")

    # # Resnet Model Results
    # # Original Image
    # axs[1, 0].imshow(image_tensor.cpu().squeeze().permute(1, 2, 0))
    # axs[1, 0].set_title(f'Original (Resnet)')
    # axs[1, 0].axis("off")

    # # Heatmap
    # heat_map_resnet = torch.nn.functional.interpolate(
    #             segm_map.unsqueeze(0),
    #             size=(224, 224),
    #             mode='bilinear'
    #         )
    # axs[1, 1].imshow(heat_map_resnet.cpu().squeeze().numpy(), cmap='jet')
    # axs[1, 1].set_title(f'Heat Map (Resnet)')
    # axs[1, 1].axis("off")

    # # Mask
    # axs[1, 2].imshow((heat_map_resnet.cpu().squeeze().numpy() > threshold_ResnetAE), cmap='gray')
    # axs[1, 2].set_title('Segmentation map (Resnet)')
    # axs[1, 2].axis("off")

    # st.pyplot(fig)
    
    
    


