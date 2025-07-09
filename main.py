# main.py

from config import *
from data_loader import load_data
from model import build_colorization_model
from train import train_model
from evaluation import evaluate_model, evaluate_pixel_similarity
from visualization import test_model

# Load data
X, Y = load_data(color_dataset_path, gray_scale_dataset_path, class_folders, size=(64, 64))
X_train, X_val = X[:int(0.8*len(X))], X[int(0.8*len(X)):]
Y_train, Y_val = Y[:int(0.8*len(Y))], Y[int(0.8*len(Y)):]

# Build model
model = build_colorization_model(input_shape)

# Train model
train_model(model, X_train, Y_train, X_val, Y_val, batch_size, epochs, weights_path)

# Evaluate
evaluate_model(model, X_val, Y_val, display_interval=200)
evaluate_pixel_similarity(model, X_val, Y_val, display_interval=400)

# Visualize
test_model(model, X_val, Y_val, index=1)
test_model(model, X_val, Y_val, index=2)

# Save model
model.save("colorization_model.keras")
print("Model saved successfully...")
