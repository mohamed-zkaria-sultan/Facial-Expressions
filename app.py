import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model2.keras')  # Replace with your model path
    return model

# Preprocess the image
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((48, 48))  # Resize to (48, 48)
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Predict the emotion
def predict_emotion(image, model, class_labels):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predictions = predictions[0]  # Remove batch dimension
    return {label: prob for label, prob in zip(class_labels, predictions)}

# Class labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the model
model = load_model()

# Streamlit app
st.title("Emotion Detection from Facial Expressions")

st.write("Upload a grayscale facial image (48x48), and the model will predict the emotion.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    # Perform prediction
    st.write("Predicting...")
    predictions = predict_emotion(image, model, class_labels)

    # Display predictions
    st.write("### Prediction:")
    highest_label = max(predictions, key=predictions.get)
    highest_probability = predictions[highest_label]
    st.write(f"{highest_label}")
