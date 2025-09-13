import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import io

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model class
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

# Class names
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

# Language selector
language = st.selectbox("Choose Language / Zaɓi Yare / Yan aṣayan Ẹ̀dé", ["English", "Hausa", "Yoruba"])

# Translations dictionary (same as original)
translations = {
    "title": {
        "English": "Cowpea Leaf Disease Detection System",
        "Hausa": "Tsarin Gano Cututtukan Ganyen Wake",
        "Yoruba": "Ero Asawari Àìsàn Ewé Èwà"
    },
    "upload_label": {
        "English": "Upload a Cowpea leaf image below to detect possible disease.",
        "Hausa": "Loda hoton ganye a ƙasa don gano yiwuwar cuta.",
        "Yoruba": "Dakun safihan aworan ewe si isalẹ lati ri àìsàn tó le wáyé."
    },
    "button": {
        "English": "Detect Disease",
        "Hausa": "Gano Cuta",
        "Yoruba": "Ṣàwárí Àìsàn"
    },
    "prediction": {
        "English": "Prediction",
        "Hausa": "Hasashe",
        "Yoruba": "Ìtẹ̀numọ̀"
    },
    "confidence": {
        "English": "Confidence",
        "Hausa": "Amincewa",
        "Yoruba": "Ìgboyà"
    },
    "about": {
        "English": "This system uses AI to detect diseases in cowpea and bean leaves. Upload an image to get a prediction.",
        "Hausa": "Wannan tsarin yana amfani da AI don gano cututtukan ganyen wake da wake. Loda hoto don samun hasashe.",
        "Yoruba": "Eto yi n lo AI lati mọ àìsàn ninu ewe èwà ati ewe èpà. Safihan aworan lati gba ìtẹnumọ."
    }
}

# Treatment class mapping (same as original)
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

# Add extra translations for treatments (same as original)
translations.update({
    "treatment_healthy": {
        "English": "The leaf appears healthy. No treatment is needed.",
        "Hausa": "Ganyen ya bayyana lafiyayye. Babu buƙatar magani.",
        "Yoruba": "Ewé náà dàbí ẹni pé kó ní àìsàn. Kò sí ìtọ́jú tó jẹ́ dandan."
    },
    "treatment_cercospora": {
        "English": "Use fungicides and remove infected leaves to prevent spread.",
        "Hausa": "Yi amfani da magungunan fungicide kuma cire ganyen da cutar ta kama.",
        "Yoruba": "Lo oogun-ifinko àti yọ àwọn ewé tó ní àìsàn kúrò láti dènà kaakiri."
    },
    "treatment_rust": {
        "English": "Apply sulfur-based fungicides and rotate crops annually.",
        "Hausa": "Yi amfani da fungicide mai sinadarin sulfur kuma canja amfanin gona kowace shekara.",
        "Yoruba": "Lo oogun-ifinko tí ó ní sulfur àti máa yí irugbin padà lọ́dọọdún."
    },
    "treatment_other": {
        "English": "Consult an agricultural extension officer for proper diagnosis.",
        "Hausa": "Tuntuɓi jami'in wayar da kan manoma don tantancewa daidai.",
        "Yoruba": "Lo ri eleto idanmoran fun ise agbe fun ayewo finifini."
    }
})

# Image transformation
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ Add background styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #26547C;
        color: #fcfcfc;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title(translations["title"][language])
st.write(translations["about"][language])
st.write(" ")
st.write(translations["upload_label"][language])

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, caption="Preview", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with col2:
        if st.button(translations["button"][language]):
            with st.spinner("Analyzing image..."):
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
                    predicted_idx = np.argmax(probabilities)
                    predicted_class = class_names[predicted_idx]
                    confidence = probabilities[predicted_idx] * 100

                st.success(f"{translations['prediction'][language]}: **{predicted_class}**")
                st.info(f"{translations['confidence'][language]}: {confidence:.2f}%")

                treatment_key = treatment_mapping.get(predicted_class, 'treatment_other')
                treatment = translations.get(treatment_key, {}).get(language, "No treatment info.")
                st.warning(treatment)

                # ✅ Download prediction as text file
                report = f"""
                {translations['prediction'][language]}: {predicted_class}
                {translations['confidence'][language]}: {confidence:.2f}%
                Treatment: {treatment}
                """
                buf = io.BytesIO()
                buf.write(report.encode())
                buf.seek(0)
                st.download_button("Download Report", data=buf, file_name="prediction.txt", mime="text/plain")
