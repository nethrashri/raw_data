import numpy as np
import tensorflow as tf
import sys
import time
from datetime import datetime
import os

def debug_model_inference(timestamp_int, model_path="wifi_app_predictor.tflite"):
    """
    Debug TFLite model inference to verify it's processing inputs correctly
    """
    print("\n" + "="*70)
    print(f"DEBUGGING INFERENCE FOR TIMESTAMP: {timestamp_int}")
    print(f"Human readable time: {datetime.fromtimestamp(timestamp_int).strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 1. Verify model file exists
    print(f"\n1. Checking model file: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} does not exist!")
        return
    print(f"Model file exists: {os.path.getsize(model_path)} bytes")

    # 2. Load the model
    print("\n2. Loading TFLite model...")
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("Model loaded successfully")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    
    # 3. Get model details
    print("\n3. Model details:")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    # 4. Examine some model tensors to check if weights exist
    print("\n4. Examining model tensors (sampling a few):")
    tensor_details = interpreter.get_tensor_details()
    weight_tensors = [t for t in tensor_details if 'weight' in t['name'].lower()]
    
    if weight_tensors:
        print(f"Found {len(weight_tensors)} weight tensors. Sampling first few:")
        for i, tensor in enumerate(weight_tensors[:3]):  # Show first 3 weight tensors
            t = interpreter.get_tensor(tensor['index'])
            non_zero = np.count_nonzero(t)
            total = np.prod(t.shape)
            print(f"  - Tensor '{tensor['name']}': shape={t.shape}, dtype={t.dtype}")
            print(f"    Non-zero values: {non_zero}/{total} ({non_zero/total:.2%})")
            print(f"    Sample values: {t.flatten()[:5]}...")  # Show first 5 values
    else:
        print("No weight tensors found - this is unusual!")
    
    # 5. Prepare input
    print("\n5. Preparing input:")
    input_shape = input_details[0]['shape']
    print(f"Expected input shape: {input_shape}")
    
    input_data = np.array([[timestamp_int]], dtype=np.float32)
    print(f"Input data: {input_data} with shape {input_data.shape}")
    
    # 6. Run inference with timing
    print("\n6. Running inference:")
    start_time = time.time()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.4f} seconds")
    
    # 7. Get and analyze output
    print("\n7. Analyzing output:")
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"Output shape: {output.shape}")
    print(f"Raw output values: {output[0]}")
    
    # Check if output has variation
    if np.all(output[0] == output[0][0]):
        print("WARNING: All output values are identical!")
    else:
        print(f"Output variation: min={np.min(output):.6f}, max={np.max(output):.6f}")
    
    # 8. Get sorted predictions
    print("\n8. Sorted predictions:")
    sorted_indices = np.argsort(output[0])[::-1]
    print(f"Class IDs in order of probability: {sorted_indices}")
    
    # 9. Try to load application names if available
    print("\n9. Looking up application names:")
    if os.path.exists('application_encoding.json'):
        try:
            import json
            with open('application_encoding.json', 'r') as f:
                encodings = json.load(f)
                print("Top 3 predictions:")
                for i, idx in enumerate(sorted_indices[:3]):
                    app_name = encodings['int_to_app'].get(str(idx), f"Unknown App {idx}")
                    prob = output[0][idx]
                    print(f"  {i+1}. {app_name} (Class ID: {idx}) - Probability: {prob:.6f}")
        except Exception as e:
            print(f"Error loading application names: {e}")
    else:
        print("No application_encoding.json file found")
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETE")
    print("="*70)
    
    return sorted_indices

if __name__ == "__main__":
    # Get timestamp from command line or use current time
    if len(sys.argv) > 1:
        timestamp = int(sys.argv[1])
    else:
        timestamp = int(time.time())
    
    # Get model path from command line if provided
    model_path = "wifi_app_predictor.tflite"
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    
    # Run debug inference
    sorted_indices = debug_model_inference(timestamp, model_path)
    
    # Print final result in simple format for easy parsing
    print("\nFinal result (space-separated class IDs):")
    print(" ".join(map(str, sorted_indices)))