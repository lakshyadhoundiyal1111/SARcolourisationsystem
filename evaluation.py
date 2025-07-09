# evaluation.py

import numpy as np
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from model import build_colorization_model  # ✅ Real model import


def evaluate_model(model, X_val, Y_val, display_interval=2):
    print("🔍 Model evaluation started...")
    total_ssim, total_psnr = 0.0, 0.0
    total_time = 0.0
    num_samples = len(X_val)

    for i in range(num_samples):
        try:
            grayscale_image = np.expand_dims(X_val[i], axis=0)
            true_color_image = Y_val[i]

            start_time = time.time()
            predicted_image = model.predict(grayscale_image, verbose=0)[0]
            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            ssim_value = ssim(true_color_image, predicted_image, data_range=1.0, win_size=7, channel_axis=-1)
            psnr_value = psnr(true_color_image, predicted_image, data_range=1.0)

            total_ssim += ssim_value
            total_psnr += psnr_value

            if (i + 1) % display_interval == 0 or (i + 1) == num_samples:
                print(f"[{i + 1}/{num_samples}] SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.2f} dB, Time: {elapsed_time:.4f}s")

        except Exception as e:
            print(f"⚠️ Error at image {i}: {e}")

    avg_ssim = total_ssim / num_samples
    avg_psnr = total_psnr / num_samples
    avg_time = total_time / num_samples

    print("\n✅ Evaluation Complete")
    print(f"📊 Average SSIM: {avg_ssim:.4f}")
    print(f"📊 Average PSNR: {avg_psnr:.2f} dB")
    print(f"⏱️ Average Inference Time: {avg_time:.4f} seconds/image")


if __name__ == "__main__":
    print("🚀 Initializing model and loading weights...")
    input_shape = (64, 64, 1)  # match your training input
    model = build_colorization_model(input_shape)
    model.load_weights("colorization_model_weights.weights.h5")

    print("📦 Generating dummy validation data...")
    X_val = np.random.rand(10, 64, 64, 1).astype("float32")
    Y_val = np.repeat(X_val, 3, axis=-1)

    print(f"🖼️ Validation data shape: {X_val.shape} (X), {Y_val.shape} (Y)")
    evaluate_model(model, X_val, Y_val)
