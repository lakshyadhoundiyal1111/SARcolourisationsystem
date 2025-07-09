import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the model
model = load_model("colorization_model.keras")

# Load preprocessed grayscale image directly
image_path = os.path.join("test", "input.jpg")
gray_input = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check shape
if gray_input.shape != (64, 64):
    raise ValueError(f"Expected image shape (64, 64), got {gray_input.shape}")

# Convert to float32 in [0, 1]
gray_input = gray_input.astype("float32") / 255.0

# Reshape to (1, 64, 64, 1)
input_img = gray_input.reshape(1, 64, 64, 1)

# Predict
output = model.predict(input_img)[0]
output_img = (output * 255).astype(np.uint8)

# Save the output image
output_path = os.path.join("test", "output.jpg")
cv2.imwrite(output_path, output_img)
print(f"Saved colorized image to {output_path}")

# Display input and output
plt.subplot(1, 2, 1)
plt.title("Grayscale Input")
plt.imshow(gray_input, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Colorized Output")
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
