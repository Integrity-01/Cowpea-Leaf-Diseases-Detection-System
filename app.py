import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Cowpea Leaf Disease Detection",
    layout="centered"
)

# ✅ Add background image using correct CSS selectors
st.markdown(
    """
    <style>
        /* Background image on whole app */
        .stApp {
            background-image: url("https://upload.wikimedia.org/wikipedia/commons/d/d5/Cowpea_leaf_showing_disease_symptoms.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }

        /* Optional: add background blur and white overlay to content */
        .main > div {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
        }

        /* Hide footer and header if you want */
        footer, header {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Main App Content
st.title("🌿 Cowpea Leaf Disease Detection App")

st.markdown("""
Welcome to the **Cowpea Leaf Disease Detection App** powered by Machine Learning and Computer Vision.

Upload a clear image of a cowpea leaf, and our model will analyze and detect possible diseases affecting the leaf.
""")

st.header("📤 Upload Cowpea Leaf Image")

uploaded_file = st.file_uploader("Choose a cowpea leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Cowpea Leaf Image", use_column_width=True)
    
    st.success("✅ Image uploaded successfully!")
    
    st.subheader("🧠 Prediction Result")
    st.info("🔍 Model is analyzing the image... (simulation)")
    
    # Simulated prediction result
    st.write("📌 **Predicted Disease:** Septoria Leaf Spot")

st.markdown("---")
st.caption("Developed by Yusuff Abdulmalik Olaitan — Integrity © 2025")
