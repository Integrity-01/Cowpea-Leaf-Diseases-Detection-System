import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model class (must match the trained architecture)
class CowpeaCNN(nn.Module):
    def __init__(self, num_classes):
        super(CowpeaCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 8)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Define class names
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

print("This is the class numbers:", len(class_names))

# Disease translations and treatment advice
translations = {
    'bean_blight': {
        'English': ("Bean Blight", "Remove infected plants. Use copper-based fungicide."),
        'Yoruba': ("Aisan Irẹ̀jẹ̀ Ewa", "Yọ awọn irugbin ti o ní àrùn. Lò oogun fungicide tó ní kòpà."),
        'Hausa': ("Cutar Blight a Wake", "Cire tsirrai masu cuta. Yi amfani da maganin fungus mai ɗauke da tagulla.")
    },
    'bean_fresh': {
        'English': ("Healthy Bean Leaf", "No disease detected."),
        'Yoruba': ("Ewe Ewa Tó Láàlááfia", "Ko si àrùn kankan."),
        'Hausa': ("Ganyen Wake Lafiya", "Ba a gano wata cuta ba.")
    },
    'bean_mosaic': {
        'English': ("Bean Mosaic Virus", "Control aphids. Use virus-free seeds."),
        'Yoruba': ("Àrùn Mosaiki Ewa", "Ṣàkóso aphid. Lò irugbin tí kò ní àrùn."),
        'Hausa': ("Cutar Mosaic a Wake", "Kula da kwari. Yi amfani da iri mara cuta.")
    },
    'bean_rust': {
        'English': ("Bean Rust", "Apply appropriate fungicides. Rotate crops."),
        'Yoruba': ("Ìbàjẹ̀ Eruku Ewa", "Lò oogun àrùn tó tọ́. Yí ọ̀nà ìṣèdá irugbin padà."),
        'Hausa': ("Rashin Lafiya a Wake (Rust)", "Yi amfani da maganin fungus. Sauya shuka lokaci-lokaci.")
    },
    'cowpea_bacterial_wilt': {
        'English': ("Cowpea Bacterial Wilt", "Remove affected plants. Ensure proper drainage."),
        'Yoruba': ("Ìgbó Cowpea Tó Ní Àrùn Bakteria", "Yọ ewébì tí a fọ́. Rí dájú pé omi ń yọ dáadáa."),
        'Hausa': ("Rashin Lafiya Bakteriya a Wake", "Cire shukar da cuta ta kama. Tabbatar da fitar ruwa sosai.")
    },
    'cowpea_fresh': {
        'English': ("Healthy Cowpea Leaf", "No disease detected."),
        'Yoruba': ("Ewe Cowpea Tó Dáa", "Ko si àrùn kankan."),
        'Hausa': ("Ganyen Wake Mai Lafiya", "Ba a gano wata cuta ba.")
    },
    'cowpea_mosaic': {
        'English': ("Cowpea Mosaic Virus", "Use resistant varieties. Control insect vectors."),
        'Yoruba': ("Àrùn Mosaiki Cowpea", "Lo irugbin tó ní àgbo. Ṣàkóso kokoro tó ń gbé àrùn."),
        'Hausa': ("Cutar Mosaic a Wake", "Yi amfani da iri masu juriya. Kula da kwari masu yada cuta.")
    },
    'cowpea_septoria': {
        'English': ("Septoria Leaf Spot", "Use fungicide and avoid overhead watering."),
        'Yoruba': ("Àpò Ewé Septoria", "Lo oogun àrùn. Má fi omi yí ewé lórí."),
        'Hausa': ("Tabon Ganye na Septoria", "Yi amfani da maganin fungus. Guji zuba ruwa daga sama.")
    }
}



# Load the model
model = CowpeaCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load('cowpea_model.pth', map_location=device))
model.eval()

# Streamlit UI
st.set_page_config(page_title="Cowpea Leaf Disease Detection", layout="centered")
st.title("🌱 Cowpea Leaf Disease Detector")
language = st.selectbox("🌐 Choose Language", ["English", "Yoruba", "Hausa"])
uploaded_file = st.file_uploader("📷 Upload a Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]

    disease_name, treatment = translations[label][language]

    #st.markdown(f"### 🧪 Detected Disease: **{disease_name}**")
    #st.markdown(f"### 💊 Treatment Advice: _{treatment}_")
