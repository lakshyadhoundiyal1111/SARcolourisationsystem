# train.py

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

def train_model(model, X_train, Y_train, X_val, Y_val, batch_size, epochs, weights_path):
    model.compile(optimizer=Adam(), loss=MeanSquaredError())
    
    try:
        history = model.fit(
            X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, Y_val)
        )
        model.save_weights(weights_path)
        print(f"Model weights saved successfully to {weights_path}.")
        return history
    except Exception as e:
        print(f"Error during training: {e}")
        return None
