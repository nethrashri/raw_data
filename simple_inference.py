import numpy as np
import tensorflow as tf
import sys

def predict_top_apps(timestamp_int, model_path="wifi_app_predictor.tflite"):

    """Predict application types and return class IDs ordered by probability"""

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