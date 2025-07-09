# config.py

color_dataset_path = './dataset/EuroSAT'
gray_scale_dataset_path = './dataset/EuroSAT_gray_scale'

class_folders = [
    'River', 'SeaLake', 'Residential', 'Pasture', 'AnnualCrop',
    'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'PermanentCrop'
]

input_shape = (64, 64, 1)
batch_size = 16
epochs = 50
weights_path = 'colorization_model_weights.weights.h5'
model_save_path = "colorization_model.keras"
