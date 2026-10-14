# ==============================================================================
# Image Feature Extraction using Pretrained ResNet18 and 2D PCA Visualization
# ==============================================================================

# 1. Imports
import torch                                         # PyTorch core library for tensor operations
import torchvision.transforms as T                   # Image transformation utilities (scaling, normalization)
from torchvision.datasets import CIFAR10            # CIFAR-10 dataset downloader & loader
from torchvision.models import resnet18, ResNet18_Weights  # Pretrained ResNet-18 architecture and weights
from sklearn.decomposition import PCA               # Principal Component Analysis for dimensionality reduction
import matplotlib.pyplot as plt                      # Matplotlib for generating 2D scatter plots

# ------------------------------------------------------------------------------
# 2. Load Pretrained CNN Model as Feature Extractor
# ------------------------------------------------------------------------------
print("Loading pretrained ResNet18 model...")

# Load ResNet-18 initialized with standard ImageNet pre-trained weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Replace the final fully connected classification layer (fc) with an Identity layer.
# This strips the 1000-class output head and yields the raw 512-dimensional feature vector.
model.fc = torch.nn.Identity()

# Set model to evaluation mode (disables dropout and freezes batch-norm running statistics)
model.eval()

# ------------------------------------------------------------------------------
# 3. Load CIFAR-10 Dataset ("Real" Images)
# ------------------------------------------------------------------------------
print("Downloading CIFAR10 dataset...")

# Define standard image preprocessing: convert PIL image to tensor, then normalize using ImageNet stats
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Automatically download and load the CIFAR-10 test partition into the ./data directory
dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)

# ------------------------------------------------------------------------------
# 4. Prepare Real Images and Generate Synthetic "Fake" Images
# ------------------------------------------------------------------------------
print("Preparing real images and generating fake noisy images...")

# Select the first 200 real images from CIFAR-10 and stack them into a single tensor of shape (200, 3, 32, 32)
real_imgs = torch.stack([dataset[i][0] for i in range(200)])

# Create 200 "fake" images by adding Gaussian (normal) random noise to the real images
fake_imgs = real_imgs + torch.randn_like(real_imgs) * 0.2

# Concatenate real and fake images along the batch dimension into a single tensor of shape (400, 3, 32, 32)
all_imgs = torch.cat([real_imgs, fake_imgs], dim=0)

# ------------------------------------------------------------------------------
# 5. Extract Feature Embeddings
# ------------------------------------------------------------------------------
print("Extracting features with ResNet18...")

# Disable gradient calculation to save memory and accelerate inference
with torch.no_grad():
    # Pass all 400 images through ResNet18 to obtain a 512-dimensional feature vector per image; convert to numpy array
    features = model(all_imgs).numpy()  # Output shape: (400, 512)

# ------------------------------------------------------------------------------
# 6. Dimensionality Reduction with PCA (512D -> 2D)
# ------------------------------------------------------------------------------
print("Running PCA to reduce features to 2D...")

# Initialize PCA to project the 512-dimensional features down to 2 principal components
pca = PCA(n_components=2)

# Fit PCA on the feature matrix and project down to 2D coordinates; shape becomes (400, 2)
features_2d = pca.fit_transform(features)

# ------------------------------------------------------------------------------
# 7. Visualize and Save the 2D PCA Plot
# ------------------------------------------------------------------------------
print("Plotting features...")

# Create a figure for plotting
plt.figure(figsize=(8, 6))

# Plot the real image features (indices 0 to 199) in blue
plt.scatter(features_2d[:200, 0], features_2d[:200, 1], c="blue", label="Real", alpha=0.6)

# Plot the fake image features (indices 200 to 399) in red
plt.scatter(features_2d[200:, 0], features_2d[200:, 1], c="red", label="Fake", alpha=0.6)

# Add title, axis labels, legend, and layout formatting
plt.title("ResNet18 Features Visualized with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.tight_layout()

# Save the visualization to disk as pca_plot.png
plt.savefig("pca_plot.png")
print("Saved visualization to pca_plot.png")

# Display the interactive plot window
plt.show()

print("Done!")
