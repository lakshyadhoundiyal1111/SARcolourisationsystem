# evaluation.py

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import accuracy_score

def evaluate_model(model, X_val, Y_val, display_interval=10):
    total_ssim, total_psnr = 0, 0
    num_samples = len(X_val)

    for i in range(num_samples):
        grayscale_image = np.expand_dims(X_val[i], axis=0)
        true_color_image = Y_val[i]

        predicted_image = model.predict(grayscale_image, verbose=0)[0]
        min_dim = min(predicted_image.shape[:2])
        win_size = min(7, min_dim)

        ssim_value = ssim(true_color_image, predicted_image, data_range=1, win_size=win_size, channel_axis=-1)
        psnr_value = psnr(true_color_image, predicted_image, data_range=1)

        total_ssim += ssim_value
        total_psnr += psnr_value

        if (i + 1) % display_interval == 0 or (i + 1) == num_samples:
            print(f"Processed {i + 1}/{num_samples} images...")

    avg_ssim = total_ssim / num_samples
    avg_psnr = total_psnr / num_samples

    print(f"\nEvaluation Complete ✅")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")

def pixel_by_pixel_similarity(predicted_image, true_image):
    predicted_image = np.round(predicted_image).astype(int)
    true_image = np.round(true_image).astype(int)

    return accuracy_score(true_image.flatten(), predicted_image.flatten()) * 100

def evaluate_pixel_similarity(model, X_val, Y_val, display_interval=100):
    total_similarity = 0
    total_images = len(X_val)

    for i in range(total_images):
        grayscale_image = X_val[i:i+1]
        true_color_image = Y_val[i]
        predicted_image = model.predict(grayscale_image, verbose=0)[0]

        similarity = pixel_by_pixel_similarity(predicted_image, true_color_image)
        total_similarity += similarity

        if (i + 1) % display_interval == 0 or (i + 1) == total_images:
            print(f"Processed {i + 1}/{total_images} images...")

    avg_similarity = total_similarity / total_images
    print(f"Average Pixel-by-Pixel Similarity: {avg_similarity:.2f}% ✅")
