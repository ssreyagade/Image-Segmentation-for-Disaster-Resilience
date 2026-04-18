from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import onnxruntime as ort

app = Flask(__name__)

# Folders
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Class labels
DISASTER_CLASSES = ['flood', 'fire', 'landslide']
CLASS_LABELS = ['Background', 'Buildings', 'Vegetation', 'Water', 'Roads', 'Vehicles']

# ONNX Sessions
clf_session = ort.InferenceSession("models/disaster_classifier.onnx")
clf_input_name = clf_session.get_inputs()[0].name

seg_sessions = {
    'flood': ort.InferenceSession("models/floodnet_unet.onnx"),
    'fire': ort.InferenceSession("models/fire_unet.onnx"),
    'landslide': ort.InferenceSession("models/landslide_unet.onnx")
}
seg_input_name = seg_sessions['flood'].get_inputs()[0].name  # assuming all have same input shape

# Preprocess
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB').resize((256, 256))
    img_np = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(img_np, axis=0), img_np

def count_objects(mask, class_id):
    labeled = label(mask == class_id)
    return len(regionprops(labeled))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/segment', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    result_img = None
    error = None
    show_card = False
    object_counts = {}

    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            error = "Please upload an image."
        else:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            img_input, img_for_seg = preprocess_image(filepath)

            # Classification
            clf_output = clf_session.run(None, {clf_input_name: img_input})
            pred_class = np.argmax(clf_output[0])
            prediction = DISASTER_CLASSES[pred_class]

            if prediction in seg_sessions:
                seg_output = seg_sessions[prediction].run(None, {seg_input_name: img_input})[0]
                pred_mask = np.argmax(seg_output[0], axis=-1)

                # Save segmentation result
                result_filename = f"seg_{filename}"
                mask_path = os.path.join(RESULT_FOLDER, result_filename)
                plt.figure(figsize=(8, 6))
                plt.imshow(pred_mask, cmap='tab20')
                plt.axis('off')
                plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                result_img = f"results/{result_filename}"

                # Count objects
                object_counts = {
                    'Buildings': count_objects(pred_mask, 1),
                    'Vegetation': count_objects(pred_mask, 2),
                    'Water': count_objects(pred_mask, 3),
                    'Roads': count_objects(pred_mask, 4),
                    'Vehicles': count_objects(pred_mask, 5)
                }

                # Pie chart
                pixel_counts = [np.sum(pred_mask == i) for i in range(1, 6)]
                labels = list(object_counts.keys())
                colors = ['#3498db', '#2ecc71', '#5dade2', '#f39c12', '#95a5a6']

                plt.figure(figsize=(6, 6))
                plt.pie(
                    pixel_counts,
                    labels=labels,
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=140,
                    textprops={'fontsize': 14}
                )
                plt.title('Object Distribution', fontsize=18)
                pie_path = os.path.join(RESULT_FOLDER, "pie_chart.png")
                plt.savefig(pie_path)
                plt.close()

                show_card = True

    return render_template("segment.html", prediction=prediction, filename=filename,
                           result_img=result_img, error=error, show_card=show_card,
                           object_counts=object_counts)

if __name__ == '__main__':
    app.run(debug=True)
