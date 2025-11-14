"""
Google Colab Setup Script
Run this in a Colab cell to set up the environment
"""

# Install dependencies
!pip install -q deepface opencv-python numpy tensorflow tf-keras

# Create necessary directories
import os
os.makedirs('dataset', exist_ok=True)
os.makedirs('utils', exist_ok=True)

print("Setup complete! Now upload your project files.")

