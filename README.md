# 🎨 Satellite Image Colorization using Deep Learning

## 📌 Overview
This project focuses on colorizing grayscale satellite images using a convolutional neural network. It is trained on the EuroSAT dataset and predicts realistic color versions of grayscale inputs. The aim is to restore or enhance satellite images for better visualization and interpretation.

---

## 📁 Dataset

- **Source**: [EuroSAT dataset](https://github.com/phelber/eurosat)
- **Classes Used**: 10 land use categories such as River, Forest, Industrial, Residential, etc.
- **Images**: 64×64 pixel satellite images in color and grayscale
- **Input**: Grayscale image  
- **Target**: Corresponding color image

---

## 🧹 Data Preprocessing

- All images are resized to `64x64` pixels.
- Grayscale images are expanded to include a channel dimension `(64, 64, 1)`.
- Pixel values are normalized to range `[0, 1]`.
- Dataset is split into training and validation sets (80:20).

The preprocessing pipeline is defined in `data_loader.py`.

---

## 🧠 Model Architecture

The model is defined in `model.py` using Keras Functional API. It is a symmetric convolutional autoencoder with:

- Multiple convolution and max pooling layers for encoding
- Upsampling and skip connections for decoding
- Final output shape: `(64, 64, 3)` (RGB)

---

## ⚙️ Training & Evaluation

Training is handled in `train.py`:
- **Optimizer**: Adam
- **Loss**: Mean Squared Error
- **Epochs**: 50
- **Batch size**: 16

Evaluation is done using:
- **SSIM (Structural Similarity Index)**
- **PSNR (Peak Signal-to-Noise Ratio)**
- **Pixel-wise Accuracy**

Results are printed after the model finishes training.

---

## 📊 Visualization

Colorization results are visualized using `matplotlib`:
- Grayscale input
- Predicted color output
- Ground truth color image

The visualization logic is in `visualization.py`.

---

## 📁 Project Structure

```
├── config.py                 # Configuration (paths, constants)
├── data_loader.py           # Loads and preprocesses image data
├── model.py                 # CNN architecture for colorization
├── train.py                 # Model training script
├── evaluation.py            # SSIM, PSNR, and pixel similarity evaluation
├── visualization.py         # Visual comparison of results
├── main.py                  # End-to-end pipeline
├── README.md                # Project documentation
```

---

## ✅ How to Run

1. Make sure the EuroSAT dataset is downloaded and properly extracted in the specified paths.
2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the pipeline:

```bash
python main.py
```

Model weights and outputs will be saved upon completion.

---

## 🧪 Requirements

```
tensorflow
numpy
opencv-python
scikit-image
scikit-learn
matplotlib
```

(You can export this using `pip freeze > requirements.txt`)

---

## 🚀 Future Improvements

- Add GAN-based models (e.g., Pix2Pix, CycleGAN)
- Real-time colorization via Streamlit or Flask
- Dataset augmentation and transfer learning
- Higher resolution outputs

---

## 👨‍💻 Contribute as you like :)

Developed as part of a major academic project.  
Trained and evaluated on real satellite image data.  
Feel free to contribute via pull requests or issues!

---

