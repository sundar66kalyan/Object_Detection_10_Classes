# ============================================================
# OBJECT DETECTION 10 CLASSES
# IMAGE CLASSIFICATION USING MOBILENETV2
# STREAMLIT WEB APPLICATION
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import time

from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="Object Detection 10 Classes",

    page_icon="📷",

    layout="wide",

    initial_sidebar_state="expanded"

)

# ============================================================
# PROJECT TITLE
# ============================================================

st.title("📷 Object Detection - 10 Classes")

st.markdown("---")

st.write(
"""
This application classifies images into one of **10 object classes**
using a **Fine-Tuned MobileNetV2** deep learning model.

Upload an image and the model will predict the object class.
"""
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(

    BASE_DIR,

    "..",

    "model",

    "Best_Model.keras"

)

# Step 1: Changed label path to use DeepLearning_ClassNames.json
LABEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "metadata",
    "DeepLearning_ClassNames.json"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found:\n{MODEL_PATH}")
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH)

    st.success("✅ MobileNetV2 model loaded successfully")

    return model

# ============================================================
# LOAD LABELS
# ============================================================

# Step 2: Changed loading code to handle list directly
@st.cache_data
def load_labels():

    if not os.path.exists(LABEL_PATH):
        st.error(f"Label file not found:\n{LABEL_PATH}")
        st.stop()

    with open(LABEL_PATH, "r") as f:
        class_names = json.load(f)

    st.success(f"✅ Loaded {len(class_names)} class names successfully")

    return class_names

# ============================================================
# LOAD EVERYTHING
# ============================================================

model = load_model()

class_names = load_labels()

# ============================================================
# MODEL DEBUG - STEP 1
# ============================================================

st.write("========== MODEL DEBUG ==========")

st.write("Model Path")
st.code(MODEL_PATH)

st.write("Model Exists")
st.write(os.path.exists(MODEL_PATH))

st.write("First Layer")
st.write(model.layers[0].__class__.__name__)

st.write("Input Shape")
st.write(model.input_shape)

st.write("Output Shape")
st.write(model.output_shape)

# Step 3: Verify the class names
st.write("Loaded Class Names")
st.write(class_names)

# ============================================================
# IMAGE PREPROCESSING - FIXED (NO DOUBLE NORMALIZATION)
# ============================================================

IMAGE_SIZE = (224,224)

def preprocess_image(image):
    """
    Preprocess image for model prediction.
    IMPORTANT: The model has a Rescaling(1./255) layer,
    so we DO NOT divide by 255 here.
    """
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image = np.array(image, dtype=np.float32)
    
    # DO NOT divide by 255 here.
    # The model's Rescaling layer handles normalization.
    
    image = np.expand_dims(image, axis=0)
    
    return image

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.image(
        "https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png",
        width=120
    )

    st.title("Project Information")

    st.markdown("---")

    st.write("### Model")

    st.success("Fine-Tuned MobileNetV2")

    st.write("### Dataset")

    st.info("ImageNet Subset (10 Classes)")

    st.write("### Image Size")

    st.info("224 × 224 RGB")

    st.write("### Number of Classes")

    st.info("10")

    st.write("### Framework")

    st.success("TensorFlow / Keras")

    st.write("### Developer")

    st.write("Kalyana Sundar")

    st.markdown("---")

    st.write("### Classes")

    for cls in class_names:

        st.write("•", cls)

# ============================================================
# MAIN LAYOUT
# ============================================================

left_col, right_col = st.columns([1,1])

# ============================================================
# IMAGE UPLOAD
# ============================================================

with left_col:

    st.subheader("Upload Image")

    uploaded_file = st.file_uploader(

        "Choose an image",

        type=["jpg","jpeg","png"]

    )

# ============================================================
# IMAGE PREVIEW
# ============================================================

with right_col:

    st.subheader("Image Preview")

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(

            image,

            caption="Uploaded Image",

            use_container_width=True

        )

    else:

        st.info("Upload an image to begin prediction.")

# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

st.subheader("Model Details")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(

        "Model",

        "MobileNetV2"

    )

with col2:

    st.metric(

        "Classes",

        "10"

    )

with col3:

    st.metric(

        "Input Size",

        "224×224"

    )

with col4:

    st.metric(

        "Status",

        "Ready"
    )

st.markdown("---")

# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    st.markdown("---")

    predict_button = st.button(
        "🚀 Predict Image",
        use_container_width=True
    )

    if predict_button:

        with st.spinner("Predicting..."):

            start_time = time.time()

            # ---------------------------------------
            # Preprocess Image
            # ---------------------------------------

            processed_image = preprocess_image(image)

            # ============================================================
            # STEP 2: CHECK THE UPLOADED IMAGE
            # ============================================================
            
            st.write("========== IMAGE SENT TO MODEL ==========")
            
            # Display the image that will be sent to the model
            # Convert back to uint8 for display (values are 0-255 since no normalization)
            display_image = processed_image[0].astype(np.uint8)
            st.image(
                display_image,
                caption="Image Sent to Model",
                use_container_width=True
            )
            
            # Also show statistics
            st.write("Processed Image Shape:", processed_image.shape)
            st.write("Processed Image Dtype:", processed_image.dtype)
            st.write("Min Pixel:", processed_image.min())
            st.write("Max Pixel:", processed_image.max())
            
            # ---------------------------------------
            # Prediction
            # ---------------------------------------

            predictions = model.predict(
                processed_image,
                verbose=0
            )

            # ============================================================
            # STEP 3: PRINT EVERY PROBABILITY
            # ============================================================
            
            st.write("========== PREDICTION PROBABILITIES ==========")
            
            for i, p in enumerate(predictions[0]):
                st.write(f"{class_names[i]:20s} : {p:.6f}")
            
            st.write("========== PREDICTION SUMMARY ==========")

            predictions = predictions[0]

            pred_index = np.argmax(predictions)

            pred_class = class_names[pred_index]

            confidence = predictions[pred_index] * 100

            inference_time = time.time() - start_time

        st.success("Prediction Completed Successfully!")

        st.markdown("---")

        # ====================================================
        # FINAL PREDICTION
        # ====================================================

        st.subheader("🏆 Final Prediction")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(

                label="Predicted Class",

                value=pred_class

            )

        with col2:

            st.metric(

                label="Confidence",

                value=f"{confidence:.2f}%"

            )

        # ====================================================
        # INFERENCE TIME
        # ====================================================

        st.info(
            f"Inference Time : {inference_time:.4f} Seconds"
        )

        st.markdown("---")

        # ====================================================
        # TOP 3 PREDICTIONS
        # ====================================================

        st.subheader("📈 Top 3 Predictions")

        top3 = np.argsort(predictions)[::-1][:3]

        for rank, idx in enumerate(top3, start=1):

            score = predictions[idx] * 100

            st.write(
                f"### {rank}. {class_names[idx]}"
            )

            st.progress(float(predictions[idx]))

            st.write(
                f"Confidence : {score:.2f}%"
            )

            st.write("")

# ============================================================
# CLASS INFORMATION
# ============================================================

st.markdown("---")

st.subheader("📚 Supported Classes")

class_info = {
    "tench": "Freshwater fish commonly found in lakes and rivers.",
    "springer": "English Springer Spaniel dog breed.",
    "cassette_player": "Portable cassette tape player.",
    "chain_saw": "Power tool used for cutting wood.",
    "church": "Religious building used for Christian worship.",
    "french_horn": "Brass musical instrument.",
    "garbage_truck": "Vehicle used for collecting waste.",
    "gas_pump": "Fuel dispensing machine.",
    "golf_ball": "Ball used in the game of golf.",
    "parachute": "Device used for slowing descent from the air."
}

cols = st.columns(2)

items = list(class_info.items())

for i, (cls, desc) in enumerate(items):

    with cols[i % 2]:

        st.markdown(f"**{cls}**")

        st.caption(desc)

# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

st.subheader("🧠 Model Information")

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.write("**Architecture**")

    st.success("Fine-Tuned MobileNetV2")

    st.write("**Framework**")

    st.success("TensorFlow 2.x")

    st.write("**Input Shape**")

    st.success("224 × 224 × 3")

    st.write("**Number of Classes**")

    st.success("10")

with info_col2:

    st.write("**Transfer Learning**")

    st.success("ImageNet Weights")

    st.write("**Output Layer**")

    st.success("Softmax")

    st.write("**Deployment**")

    st.success("Streamlit")

    st.write("**Status**")

    st.success("Production Ready")

# ============================================================
# PERFORMANCE
# ============================================================

st.markdown("---")

st.subheader("📊 Model Performance")

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("Accuracy", "98.22%")

metric2.metric("Precision", "98.28%")

metric3.metric("Recall", "98.22%")

metric4.metric("F1 Score", "98.23%")

st.success("Best Model Selected Automatically after comparing CNN, MobileNetV2, ResNet50, EfficientNetB0, Random Forest and Linear SVM.")

# ============================================================
# TECHNOLOGIES USED
# ============================================================

st.markdown("---")

st.subheader("🛠 Technologies Used")

st.write("""
- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Pillow
- Streamlit
- Google Colab
- Google Drive
""")

# ============================================================
# SIDEBAR FOOTER
# ============================================================

with st.sidebar:

    st.markdown("---")

    st.write("### Project")

    st.write("Object Detection - 10 Classes")

    st.write("Version : 1.0")

    st.write("Deployment : Streamlit")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center'>

### Image Classification - 10 Classes

Developed using **TensorFlow**, **MobileNetV2**, and **Streamlit**

**Developer:** Kalyana Sundar

© 2026 All Rights Reserved

</div>
""",
unsafe_allow_html=True 
)