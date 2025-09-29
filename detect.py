# detect.py

from ultralytics import YOLO
import cv2
import os

# --- Configuration ---

# Path to the image you want to run detection on
# IMPORTANT: Replace 'test_image.jpg' with the actual path or filename of your image
# Make sure your image is in the 'object_detector_project' folder for simplicity
IMAGE_PATH = 'test_image.jpg' 

# We will use the 'yolov8n.pt' model, which is the 'nano' version. 
# It's small, fast, and great for starting out.
MODEL_NAME = 'yolov8n.pt' 

# --- Main Detection Function ---

def run_object_detection(image_path, model_name):
    """
    Loads a YOLO model, runs detection on an image, and displays the result.
    """
    print(f"Loading model: {model_name}...")
    
    # Load the pre-trained YOLO model
    # The first time you run this, it will automatically download the model file.
    model = YOLO(model_name)  

    if not os.path.exists(image_path):
        print(f"\nERROR: Image file not found at '{image_path}'")
        print("Please place your image in the project folder and update the IMAGE_PATH variable.")
        return

    print(f"Running detection on image: {image_path}...")
    
    # Run inference (detection) on the image
    # We set 'save=True' so YOLO automatically saves the result image
    results = model.predict(source=image_path, save=True, conf=0.5)

    print("\nDetection complete!")
    print("---")
    
    # YOLO saves the result to a 'runs/detect/...' folder. 
    # We'll print the location so you know where to find the output image.
    if results:
        # Get the path to the saved image (this is an approximation, but generally correct)
        # It's usually in 'runs/detect/predict/' or 'runs/detect/predictN/'
        try:
            output_dir = results[0].save_dir
            print(f"Result image saved successfully in: {output_dir}")
            print("Look for the image with bounding boxes drawn on it in that folder.")
        except Exception:
            print("Could not automatically determine the output folder path.")
            print("Please check the 'runs/detect' folder in your project directory.")


if __name__ == "__main__":
    # --- Instructions for testing ---
    print("-----------------------------------------------------")
    print("OBJECT DETECTOR PROJECT - YOLO")
    print("-----------------------------------------------------")
    print("BEFORE RUNNING:")
    print("1. Get a test image (e.g., a photo with people, cars, or animals).")
    print(f"2. Name it '{IMAGE_PATH}' and place it in this 'object_detector_project' folder.")
    print("3. Then run this script.")
    print("-----------------------------------------------------")
    
    run_object_detection(IMAGE_PATH, MODEL_NAME)