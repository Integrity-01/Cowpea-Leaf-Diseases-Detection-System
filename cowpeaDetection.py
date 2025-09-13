import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model definition
class CowpeaCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CowpeaCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 56 * 56, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 8)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Class labels
class_names = [
    'Bean Blight',
    'Bean Fresh Leaf',
    'Bean Mosaic Virus',
    'Bean Rust',
    'Cowpea Bacterial wilt',
    'Cowpea Fresh Leaf',
    'Cowpea Mosaic viroria leaf spot',
    'Cowpea Septoria Leaf Spot'
]

# Load model
model = CowpeaCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load('cowpea_model.pth', map_location=device))
model.eval()

# Sidebar: Language selector
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3242/3242257.png", width=80)
language = st.sidebar.selectbox("🌐 Choose Language / Zaɓi Yare / Ẹ̀dá:", ["English", "Hausa", "Yoruba"])

# Translations
translations = {
    "title": {
        "English": "🌿 Cowpea Leaf Disease Detection System",
        "Hausa": "🌿 Tsarin Gano Cututtukan Ganyen Wake",
        "Yoruba": "🌿 Ẹ̀rọ Idanimọ́ Àìsàn Ewé Èwà"
    },
    "upload_label": {
        "English": "📤 Upload a Cowpea leaf image below to detect possible disease.",
        "Hausa": "📤 Loda hoton ganye a ƙasa don gano yiwuwar cuta.",
        "Yoruba": "📤 Safihan aworan ewe si isalẹ lati ri àìsàn tó le wáyé."
    },
    "button": {
        "English": "🧪 Detect Disease",
        "Hausa": "🧪 Gano Cuta",
        "Yoruba": "🧪 Ṣàwárí Àìsàn"
    },
    "prediction": {
        "English": "🔍 Prediction",
        "Hausa": "🔍 Hasashe",
        "Yoruba": "🔍 Ìtẹ̀numọ̀"
    },
    "confidence": {
        "English": "📈 Confidence",
        "Hausa": "📈 Amincewa",
        "Yoruba": "📈 Ìgboyà"
    },
    "treatment_healthy": {
        "English": "✅ The leaf appears healthy. No treatment is needed.",
        "Hausa": "✅ Ganyen ya bayyana lafiyayye. Babu buƙatar magani.",
        "Yoruba": "✅ Ewé náà dàbí ẹni pé kó ní àìsàn. Kò sí ìtọ́jú tó jẹ́ dandan."
    },
    "treatment_cercospora": {
        "English": "💊 Use fungicides and remove infected leaves to prevent spread.",
        "Hausa": "💊 Yi amfani da magungunan fungicide kuma cire ganyen da cutar ta kama.",
        "Yoruba": "💊 Lo oogun-ifinko àti yọ àwọn ewé tó ní àìsàn kúrò láti dènà kaakiri."
    },
    "treatment_rust": {
        "English": "🧴 Apply sulfur-based fungicides and rotate crops annually.",
        "Hausa": "🧴 Yi amfani da fungicide mai sinadarin sulfur kuma canja amfanin gona kowace shekara.",
        "Yoruba": "🧴 Lo oogun-ifinko tí ó ní sulfur àti máa yí irugbin padà lọ́dọọdún."
    },
    "treatment_other": {
        "English": "👩‍🌾 Consult an agricultural extension officer for proper diagnosis.",
        "Hausa": "👩‍🌾 Tuntuɓi jami'in wayar da kan manoma don tantancewa daidai.",
        "Yoruba": "👩‍🌾 Bẹrẹ pẹ̀lú olùkọ́ni ọgbìn fun ìtọ́sọ́nà tó péye."
    }
}

treatment_mapping = {
    'Bean Fresh Leaf': 'treatment_healthy',
    'Cowpea Fresh Leaf': 'treatment_healthy',
    'Bean Blight': 'treatment_cercospora',
    'Cowpea Mosaic viroria leaf spot': 'treatment_cercospora',
    'Bean Rust': 'treatment_rust',
    'Cowpea Bacterial wilt': 'treatment_rust',
    'Cowpea Septoria Leaf Spot': 'treatment_cercospora',
    'Bean Mosaic Virus': 'treatment_other'
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Main app interface
st.markdown(f"## {translations['title'][language]}")
st.markdown("---")
st.markdown(f"### {translations['upload_label'][language]}")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("#### 📷 Uploaded Image Preview")
    st.image(image, use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    if st.button(translations["button"][language]):
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_idx = np.argmax(probabilities)
            predicted_class = class_names[predicted_idx]
            confidence = probabilities[predicted_idx] * 100

            st.markdown(f"### ✅ {translations['prediction'][language]}: **{predicted_class}**")
            st.markdown(f"### 🔢 {translations['confidence'][language]}: **{confidence:.2f}%**")

            treatment_key = treatment_mapping.get(predicted_class, 'treatment_other')
            treatment = translations.get(treatment_key, {}).get(language, "No treatment information.")

            st.markdown("---")
            st.markdown(f"### 🩺 Treatment Suggestion:")
            st.warning(treatment)


st.markdown("---")
st.caption("Developed by Yusuff Abdulmalik Olaitan — Integrity © 2025")
