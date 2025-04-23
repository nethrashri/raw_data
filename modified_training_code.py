import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
import os
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define a custom layer that generates multiple timestamp predictions
class SixHourPredictionLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.interval_minutes = 30
        self.num_samples = 12  # 6 hours with 30-minute intervals
        
        # Create dense layers for time offset predictions
        self.prediction_layers = []
        for i in range(self.num_samples + 1):
            self.prediction_layers.append(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    def call(self, inputs):
        base_timestamp = inputs
        
        # Generate predictions for each time offset
        predictions = []
        for i in range(self.num_samples + 1):
            # Calculate time offset in seconds
            offset = i * (self.interval_minutes * 60)
            # Add offset to base timestamp
            time_point = base_timestamp + tf.cast(offset, tf.float32)
            # Get prediction for this time point
            pred = self.prediction_layers[i](time_point)
            predictions.append(pred)
        
        # Stack predictions along a new axis
        stacked_preds = tf.stack(predictions, axis=1)
        
        # Average predictions across time points
        avg_pred = tf.reduce_mean(stacked_preds, axis=1)
        
        # Return the averaged prediction
        return avg_pred

# Define a simple TFLite-compatible "transformer-like" block
class TFLiteTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, ff_dim, **kwargs):
        super().__init__(**kwargs)
        
        # Feature transformation layers
        self.dense1 = tf.keras.layers.Dense(embed_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(ff_dim, activation='relu')
        self.dense3 = tf.keras.layers.Dense(embed_dim)
        
        # Normalization
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs):
        # Feature transformation
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        
        # Residual connection
        return self.norm(x + inputs)

def preprocess_dataset(input_file="custom_balanced_wifi_usage_dataset.csv", output_file="wifi_dataset.csv"):
    """Preprocess the raw dataset"""
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Convert timestamps to epoch time
    def convert_to_epoch(timestamp_str):
        try:
            # Try common formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d %H:%M:%S',
                '%d-%m-%Y %H:%M:%S',
                '%m/%d/%Y %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(timestamp_str, fmt)
                    return int(dt.timestamp())
                except ValueError:
                    continue
            
            # Try direct timestamp
            return int(float(timestamp_str))
            
        except Exception as e:
            print(f"Error converting timestamp '{timestamp_str}': {e}")
            return 0
    
    # Convert timestamps
    print("Converting timestamps...")
    df['TimestampInt'] = df['Timestamp'].apply(convert_to_epoch)
    
    # Encode application types
    print("Encoding application types...")
    app_types = df['Application_Type'].unique()
    app_to_int = {app: i for i, app in enumerate(app_types)}
    int_to_app = {str(i): app for app, i in app_to_int.items()}
    
    df['ApplicationTypeEncoded'] = df['Application_Type'].map(app_to_int)
    
    # Save encodings
    encodings = {
        'app_to_int': app_to_int,
        'int_to_app': int_to_app
    }
    
    with open('application_encoding.json', 'w') as f:
        json.dump(encodings, f, indent=2)
    
    print(f"Saved application encodings with {len(app_types)} unique types.")
    
    # Create model dataset
    model_df = df[['TimestampInt', 'ApplicationTypeEncoded']].copy()
    model_df = model_df.dropna()
    
    # Save preprocessed data
    model_df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")
    
    return model_df

def train_wifi_model(data_path="wifi_dataset.csv"):
    """Train a TFLite model to predict application types for 6 hours"""
    
    # Preprocess data if needed
    if not os.path.exists(data_path):
        preprocess_dataset()
    
    # Load the dataset
    print("Loading preprocessed dataset...")
    df = pd.read_csv(data_path)
    
    # Use raw timestamp as input
    X = df['TimestampInt'].values.reshape(-1, 1)
    y = df['ApplicationTypeEncoded'].values
    
    # Convert to float32 for better numerical stability
    X = X.astype(np.float32)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Get number of classes
    num_classes = len(np.unique(y))
    print(f"Number of application types: {num_classes}")
    
    # Build a TFLite-compatible model
    print("Building model...")
    
    # Define dimensions
    embed_dim = 64
    ff_dim = 128
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
    
    # Embedding layer
    x = tf.keras.layers.Dense(embed_dim)(inputs)
    
    # Transformer-like block
    x = TFLiteTransformerBlock(embed_dim=embed_dim, ff_dim=ff_dim)(x)
    
    # Feature extraction layers
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Print model summary
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    print("Training model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save model in TF format
    model.save('wifi_model.keras')
    print("Model saved in Keras format")
    
    # Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Ensure only built-in ops
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = False
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open('wifi_app_predictor.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("TFLite model saved successfully")
    
    # Create a simple inference function to demonstrate usage
    def predict_application(timestamp):
        interpreter = tf.lite.Interpreter(model_path="wifi_app_predictor.tflite")
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare input
        input_data = np.array([[timestamp]], dtype=np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Get sorted indices by probability (descending)
        sorted_indices = np.argsort(output[0])[::-1]
        
        return sorted_indices
    
    # Test the model with a sample timestamp
    test_timestamp = X_test[0][0]
    predicted_classes = predict_application(test_timestamp)
    
    print(f"\nTest prediction for timestamp {test_timestamp}:")
    print(f"Top class IDs in order of probability: {' '.join(map(str, predicted_classes))}")
    
    # Load encodings if available
    try:
        with open('application_encoding.json', 'r') as f:
            encodings = json.load(f)
            top_app_name = encodings['int_to_app'][str(int(predicted_classes[0]))]
            print(f"Top predicted application: {top_app_name}")
    except Exception as e:
        print(f"Could not load application encodings: {e}")
    
    print("Training and conversion complete!")
    
    return model

def create_simple_inference_script():
    """Create a simple inference script for the TFLite model"""
    code = """
import numpy as np
import tensorflow as tf
import sys

def predict_top_apps(timestamp_int, model_path="wifi_app_predictor.tflite"):

    Predict application types and return class IDs ordered by probability"""
    
    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare input: Shape must be (1, 1)
    input_data = np.array([[timestamp_int]], dtype=np.float32)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Get indices sorted by probability (highest first)
    sorted_indices = np.argsort(output[0])[::-1]
    
    return sorted_indices

if __name__ == "__main__":
    if len(sys.argv) > 1:
        timestamp = int(sys.argv[1])
    else:
        import time
        timestamp = int(time.time())
    
    # Get sorted class IDs 
    class_ids = predict_top_apps(timestamp)
    
    # Print space-separated class IDs
    print(" ".join(map(str, class_ids)))
"""
    
    with open('predict_top_apps.py', 'w') as f:
        f.write(code)
    
    print("Created simple inference script: predict_top_apps.py")

if __name__ == "__main__":
    # Train the model
    train_wifi_model()
    
    # Create simple inference script
    create_simple_inference_script()