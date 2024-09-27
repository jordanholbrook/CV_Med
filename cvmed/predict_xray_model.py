# This script performs the following steps:
# 
# 1. Defines a neural network model architecture using PyTorch:
#    - A sequential convolutional neural network (CNN) with 5 convolutional layers followed by a fully connected (FC) layer.
#    - Each convolutional layer uses ReLU activation, batch normalization, and pooling.
# 
# 2. Defines a function `preprocess_image()` to preprocess a single image:
#    - The image is opened from the specified path, resized, converted to grayscale, transformed to a tensor, and normalized.
#    - The image tensor is reshaped to include a batch dimension, as required by PyTorch models.
# 
# 3. Loads the saved model:
#    - The network architecture is instantiated with 1 input channel (grayscale) and 2 output classes (binary classification).
#    - The model's state is loaded from a file (`xray_model.pth`), which contains the trained parameters.
#    - The model is set to evaluation mode (`eval()`).
# 
# 4. Defines data transformations:
#    - The image is converted to grayscale, resized to 28x28 pixels, and normalized using the same parameters as during training.
# 
# 5. Preprocesses an image from the file path:
#    - A sample image is preprocessed using the function `preprocess_image()` and the transformations defined earlier.
# 
# 6. Runs the model to make a prediction:
#    - The preprocessed image is passed through the model in evaluation mode (without gradient calculations).
#    - The predicted class is determined from the model's output using the `torch.max()` function, which finds the class with the highest score.
# 
# 7. Outputs the predicted class to the console.


from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim


n_channels = 1
n_classes = 2
task = 'binary-class'

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
    
# Function to preprocess an individual image
def preprocess_image(image_path):
    img = Image.open(image_path)  # Open the image
    img = data_transform(img)     # Apply the transformations
    img = img.unsqueeze(0)        # Add batch dimension (1, C, H, W)
    return img

# Load the model's state_dict into the same model architecture
loaded_model = Net(in_channels=n_channels, num_classes=n_classes)  # Initialize the model architecture
loaded_model.load_state_dict(torch.load('../models/xray_model.pth'))  # Load the saved parameters
loaded_model.eval()


# Define the transformations (same as during training)
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert the image to grayscale
    transforms.Resize((28, 28)),  # Resize image to the size your model expects
    transforms.ToTensor(),        # Convert image to a tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize using the same mean and std from training
])

# Example usage with a file path to the uploaded image
image_path = '../med_images/xray/unnamed-2.jpg'  # Replace with the actual file path
preprocessed_img = preprocess_image(image_path)

# Make prediction
with torch.no_grad():
    output = loaded_model(preprocessed_img)  # Get model output
    _, predicted_class = torch.max(output, 1)  # Get predicted class

# Print the predicted class
print(f"Predicted class: {predicted_class.item()}")
