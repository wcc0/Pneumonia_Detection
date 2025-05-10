import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
from collections import defaultdict
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

# sidebar
st.sidebar.title("Model Info")
st.sidebar.markdown("**Architecture:** ResNet-34 (customized)")
st.sidebar.markdown("**Dataset:** 3 Kinds of Pneumonia (Kaggle)")
st.sidebar.markdown("**Input Size:** 224×224 grayscale")
st.sidebar.markdown("**Classes:** Normal, COVID-19, Pneumonia-Bacterial, Pneumonia-Viral")
st.sidebar.markdown("**Trained Acc:** ~95%")

label_classes = ['Normal', 'COVID-19', 'Pneumonia-Bacterial', 'Pneumonia-Viral']

# preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# load model
@st.cache_resource
def load_model():
    model = models.resnet34(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, len(label_classes))
    model.load_state_dict(torch.load("resnet34_100.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# GradCAM
cam_extractor = GradCAM(model, target_layer="layer4") # attach onto final conv layer

# inference + CAM
@st.cache_data
def process_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    tensor.requires_grad_()
    output = model(tensor)
    pred = int(torch.argmax(output, dim=1).item())
    prob = torch.softmax(output, dim=1).detach().numpy()[0]
    act_map = cam_extractor(pred, output)[0]
    pil_input = to_pil_image(transform(img))
    cam_overlay = overlay_mask(pil_input.convert("RGB"), to_pil_image(act_map, mode='F'), alpha=0.5)
    return {
        "filename": uploaded_file.name,
        "class": label_classes[pred],
        "image": img,
        "cam": cam_overlay,
        "probs": prob
    }

# application
st.title("Pneumonia X-ray Classifier with Grad-CAM")
uploaded_files = st.file_uploader("Upload chest X-rays", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.markdown("## Organized Summary by Predicted Class")

    # sort predictions into tabs
    results_by_class = defaultdict(list)
    for file in uploaded_files:
        result = process_image(file)
        results_by_class[result["class"]].append(result)

    tabs = st.tabs(results_by_class.keys())
    for tab, class_name in zip(tabs, results_by_class):
        with tab:
            entries = results_by_class[class_name]
            total = len(entries)
            st.markdown(f"### {class_name} — {total} case(s)")

            page = st.number_input(f"Page for {class_name}", min_value=0, max_value=total - 1, value=0, key=class_name)
            entry = entries[page]

            col1, col2 = st.columns(2)
            with col1:
                st.image(entry["image"], caption=f"{entry['filename']} - Original", use_container_width=True)
            with col2:
                st.image(entry["cam"], caption="Grad-CAM", use_container_width=True)

            st.markdown("### Prediction Probabilities")
            import matplotlib.ticker as mticker

            # prediction graph
            fig, ax = plt.subplots(figsize=(3, 2))

            y_pos = np.arange(len(label_classes))
            ax.barh(y_pos, entry["probs"], color='cornflowerblue', height=0.4)

            ax.set_xlim([0, 1])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(label_classes, fontsize=8)
            ax.invert_yaxis()

            for i, v in enumerate(entry["probs"]):
                ax.text(1.05, i, f"{v:.2%}".rjust(7), va='center', fontsize=8, ha='left')

            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.set_xlabel("Probability", fontsize=8)
            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=8)
            fig.subplots_adjust(left=0.3, right=0.85)

            st.pyplot(fig)


# check model path
st.text(f"Model path: resnet34_100.pth")
st.text(f"Exists: {os.path.exists('resnet34_100.pth')}")
