import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Use a cache to load the model only once
@st.cache_resource
def load_model():
    """Loads the saved Keras model."""
    try:
        model = tf.keras.models.load_model('scene_classifier_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess the uploaded image
def preprocess_image(image):
    """
    Preprocesses the image to the format the model expects.
    - Resizes to (150, 150)
    - Converts to a NumPy array
    - Rescales pixel values to [0, 1]
    - Adds a batch dimension
    """
    img = image.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array

# Define the class names in the correct order
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# --- Streamlit App ---

st.title("🏞️ Natural Scene Image Classifier")
st.write("Upload an image, and the model will predict which type of scene it is!")

# Load the model
model = load_model()

if model is None:
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image and make a prediction
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(prediction) * 100

    # Display the result
    st.success(f"Prediction: **{predicted_class_name.capitalize()}**")
    st.info(f"Confidence: **{confidence:.2f}%**")