import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# Define sigmoid function
def sigmoid(x): return 1/(1+torch.exp(-x))

# Define the model
model = nn.Sequential(
    nn.Linear(28*28, 30),
    nn.ReLU(),
    nn.Linear(30, 1)
)

# Load trained weights (placeholder, update with actual path), this is the same model used in the assignment
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Define transformation for input image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# Streamlit app
st.title("MNIST Digit Classifier (3 vs. 7)")

uploaded_file = st.file_uploader("Upload an image of a digit (3 or 7)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    image = transform(image).view(1, -1)  # Flatten image
    
    with torch.no_grad():
        output = model(image).sigmoid()
        prediction = "3" if output.item() > 0.5 else "7"
    
    st.write(f"### Prediction: {prediction}")
