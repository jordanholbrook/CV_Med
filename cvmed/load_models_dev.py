from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim


n_channels = 1
n_classes = 2
task = 'binary-class'

# Define the same transformations used during training
data_transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Ensure the image is resized to the expected input size
    transforms.ToTensor(),        # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Apply normalization (same as during training)
])

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    # Load the image
    img = Image.open(image_path)

    # Apply transformations (resize, convert to tensor, normalize)
    img = data_transform(img)

    # Add a batch dimension (1, C, H, W)
    img = img.unsqueeze(0)

    return img


class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
"""
model = Net(in_channels=n_channels, num_classes=n_classes)
    
# define loss function and optimizer
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
    
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
"""
# Load the model's state_dict into the same model architecture
loaded_model = Net(in_channels=n_channels, num_classes=n_classes)  # Initialize the model architecture
loaded_model.load_state_dict(torch.load('../models/xray_model.pth'))  # Load the saved parameters
loaded_model.eval()


# Example usage with a file path to the uploaded image
image_path = 'med_images/xray/.jpeg'  # Replace with the actual file path
preprocessed_img = preprocess_image(image_path)


# Make prediction
with torch.no_grad():
    output = loaded_model(preprocessed_img)  # Get model output
    _, predicted_class = torch.max(output, 1)  # Get predicted class

# Print the predicted class
print(f"Predicted class: {predicted_class.item()}")



import os
from PIL import Image
import torch
from torchvision import transforms

# Define the transformations (same as during training)
data_transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize image to the size your model expects
    transforms.ToTensor(),        # Convert image to a tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize using the same mean and std from training
])

# Function to preprocess an individual image
def preprocess_image(image_path):
    img = Image.open(image_path)  # Open the image
    img = data_transform(img)     # Apply the transformations
    img = img.unsqueeze(0)        # Add batch dimension (1, C, H, W)
    return img

# Directory containing the images
image_dir = 'med_images/xray/'  # Replace with the path to your image folder

# Loop through all files in the directory
for filename in os.listdir(image_dir):
    # Only process files that are images (optional, based on your file structure)
    if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
        # Full path to the image
        image_path = os.path.join(image_dir, filename)
        
        # Preprocess the image
        preprocessed_img = preprocess_image(image_path)
        
        # Make prediction using the loaded model
        loaded_model.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            output = loaded_model(preprocessed_img)  # Get model output
            _, predicted_class = torch.max(output, 1)  # Get the predicted class

        # Print the results for the current image
        print(f"Image: {filename} | Predicted class: {predicted_class.item()}")

from PIL import Image
import torch
from torchvision import transforms

# Define the transformations (same as during training)
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert the image to grayscale
    transforms.Resize((28, 28)),  # Resize image to the size your model expects
    transforms.ToTensor(),        # Convert image to a tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize using the same mean and std from training
])

# Function to preprocess an individual image
def preprocess_image(image_path):
    img = Image.open(image_path)  # Open the image
    img = data_transform(img)     # Apply the transformations
    img = img.unsqueeze(0)        # Add batch dimension (1, C, H, W)
    return img

# Example usage with a file path to the uploaded image
image_path = '../med_images/xray/unnamed-7.jpg'  # Replace with the actual file path
preprocessed_img = preprocess_image(image_path)

# Make prediction
loaded_model.eval()  # Ensure the model is in evaluation mode
with torch.no_grad():
    output = loaded_model(preprocessed_img)  # Get model output
    _, predicted_class = torch.max(output, 1)  # Get predicted class

# Print the predicted class
print(f"Predicted class: {predicted_class.item()}")


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd

# Load the saved model
loaded_model = joblib.load('../models/medical_cost_predictor_model_v3.pkl')
print("Model loaded successfully.")
# Example of how to make predictions with the loaded model
# X_new should be your new data in the same format as the training data
# For example, if you get inputs from the web app:
X_new = pd.DataFrame({
    'Age': [34],
    'Sex': ['M'],
    'Zipcode': [77231],
    'Image_Type': ['Skin'],
    'Diagnosis': ['melanoma']
})

# Make predictions with the loaded model
predicted_cost = loaded_model.predict(X_new)
print(f"Predicted medical cost: {predicted_cost[0]}")