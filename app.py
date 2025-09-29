# app.py - Flask Backend Server (Image-Only, with Bounding Box Data)

from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from ultralytics import YOLO
import os
import os.path as osp
import json 

# --- Configuration ---
app = Flask(__name__)

# Configure upload and result folders
UPLOAD_FOLDER = 'uploads'
RESULTS_BASE_FOLDER = 'runs/detect' 
MODEL_NAME = 'yolov8n.pt' 

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ONLY IMAGE EXTENSIONS ARE ALLOWED
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff', 'tif'}
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_BASE_FOLDER'] = RESULTS_BASE_FOLDER

# Load the YOLO model once when the server starts
print(f"Loading YOLO model: {MODEL_NAME}...")
MODEL = YOLO(MODEL_NAME)
print("YOLO model loaded successfully.")
# --------------------

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Handles both GET (displaying the upload page with potential errors) 
    and POST (handling the file upload and analysis).
    """
    # CRITICAL: Retrieve error message from URL parameters for display on GET request
    error_message = request.args.get('error') 
    
    if request.method == 'POST':
        # Define default values
        detection_details = [] 
        predict_folder_name = None
        original_filename = None
        
        if 'file' not in request.files:
            return redirect(url_for('upload_file', error="No file selected for upload."))
        
        file = request.files['file']
        original_filename = file.filename
        
        if file.filename == '':
            return redirect(url_for('upload_file', error="No file was chosen."))
        
        # --- ALL PROCESSING STARTS HERE ---
        if file and allowed_file(file.filename):
            # 1. Save the uploaded file
            filename = original_filename 
            file_path = osp.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            print(f"File saved: {file_path}, Type: image")
            
            # 2. Run object detection
            results = MODEL.predict(source=file_path, save=True, conf=0.25, iou=0.4)

            # --- Process Results, Count Objects, AND Gather Bounding Box Details ---
            object_stats = {} 
            
            if results and results[0].boxes:
                detections = results[0].boxes
                
                for box in detections:
                    class_id = int(box.cls[0])
                    class_name = MODEL.names.get(class_id, 'Unknown')
                    
                    confidence_value = float(box.conf[0])
                    confidence_display = round(confidence_value, 2)
                    
                    # Store detection data
                    detection_details.append({
                        'label': class_name,
                        'confidence': confidence_display,
                        'x_center': box.xywhn[0][0].item(),
                        'y_center': box.xywhn[0][1].item(),
                        'width': box.xywhn[0][2].item(),
                        'height': box.xywhn[0][3].item()
                    })

                    # Accumulate stats
                    if class_name not in object_stats:
                        object_stats[class_name] = {'count': 0, 'total_conf': 0}
                    
                    object_stats[class_name]['count'] += 1
                    object_stats[class_name]['total_conf'] += confidence_value
            # ----------------------------------------------------------------------------
            
            # 3. Finalize object counts (Calculate Average Confidence)
            final_object_counts = {}
            for name, stats in object_stats.items():
                if stats['count'] > 0:
                    avg_conf = (stats['total_conf'] / stats['count']) * 100
                    final_object_counts[name] = {
                        'count': stats['count'],
                        'avg_conf': f"{avg_conf:.1f}%" # Format as "XX.X%"
                    }
                else:
                    final_object_counts[name] = {'count': 0, 'avg_conf': "0.0%"}

            # --- FILE FINDING AND FOLDER NAME LOGIC (Robust Image Path) ---
            found_filename = filename # Default to original name
            predict_folder_name = app.config['UPLOAD_FOLDER'] # Default fallback folder

            if results and results[0].save_dir:
                result_dir = results[0].save_dir
                yolo_folder_name = osp.basename(result_dir) 
                
                # Check for the file inside the YOLO output folder
                output_path = osp.join(result_dir, filename)
                
                if osp.exists(output_path):
                    predict_folder_name = yolo_folder_name # Success: use the YOLO folder
                    found_filename = filename
                else:
                    print(f"Warning: Processed file not found in {result_dir}. Falling back to original image.")

            # 4. Return the result page
            print(f"Serving result from folder: {predict_folder_name} with file {found_filename}")
            return render_template('result.html', 
                                    predict_folder=predict_folder_name, 
                                    filename=found_filename, 
                                    object_counts=final_object_counts, # Pass the detailed stats
                                    detection_details_json=json.dumps(detection_details))
            
        
        # If file validation failed (i.e., not an allowed file type), redirect home
        file_ext = original_filename.rsplit('.', 1)[1].upper() if '.' in original_filename and '.' != original_filename[-1] else 'Unknown'
        error_msg = f"Invalid file type. Only image formats ({', '.join(sorted(ALLOWED_EXTENSIONS))}) are allowed. You uploaded a .{file_ext} file."
        return redirect(url_for('upload_file', error=error_msg))
        
    # For GET request (initial page load):
    return render_template('index.html', error_message=error_message) # Pass error message to template

@app.route('/results/<predict_folder>/<filename>')
def show_result(predict_folder, filename):
    """
    This route serves the processed image file from the results folder 
    OR the original file from the uploads folder (if detection failed/no detections).
    """
    
    # CRITICAL FIX: Check if the request is for the fallback upload folder
    if predict_folder == app.config['UPLOAD_FOLDER']:
        # Serve original, unprocessed image from the main upload directory
        base_dir = app.config['UPLOAD_FOLDER']
    else:
        # Serve the processed image from the YOLO results directory
        base_dir = osp.join(app.config['RESULTS_BASE_FOLDER'], predict_folder)

    # --- Get MIME type for files (Image Only) ---
    mimetype = 'application/octet-stream' 
    
    if '.' in filename:
        file_ext = filename.rsplit('.', 1)[1].lower()
    else:
        file_ext = ''
    
    # Include all allowed image types for MIME type setting
    if file_ext in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'tiff', 'tif']:
        mimetype = f'image/{file_ext}'
        if file_ext in ['tif', 'tiff']: 
             mimetype = 'image/tiff'

    # Check if the file exists before attempting to serve it
    if not osp.exists(osp.join(base_dir, filename)):
        print(f"Error: File not found at {osp.join(base_dir, filename)}")
        return "Image Not Found", 404
        
    return send_from_directory(base_dir, filename, mimetype=mimetype)


if __name__ == '__main__':
    app.run(debug=True)