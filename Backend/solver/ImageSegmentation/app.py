from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import subprocess
from PIL import Image
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# (rest of your Flask code)


# Load the trained SVM model and scaler
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def rgb_of_pixel(im, x, y):
    r, g, b = im.getpixel((x, y))
    return (r, g, b)

def average_rgb_in_circle(im, center_x, center_y, radius):
    rgb_values = []
    for x in range(center_x - radius, center_x + radius + 1):
        for y in range(center_y - radius, center_y + radius + 1):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                rgb_values.append(rgb_of_pixel(im, x, y))
    if rgb_values:
        avg_rgb = np.mean(rgb_values, axis=0)
        return avg_rgb
    else:
        return (0, 0, 0)

def get_center_coordinates(width, height):
    center_width = width // 2
    center_height = height // 2
    face_edge_distance = min(width, height) // 3
    centers = [
        (center_width - face_edge_distance, center_height - face_edge_distance),
        (center_width, center_height - face_edge_distance),
        (center_width + face_edge_distance, center_height - face_edge_distance),
        (center_width - face_edge_distance, center_height),
        (center_width, center_height),
        (center_width + face_edge_distance, center_height),
        (center_width - face_edge_distance, center_height + face_edge_distance),
        (center_width, center_height + face_edge_distance),
        (center_width + face_edge_distance, center_height + face_edge_distance)
    ]
    return centers

def classify_color(rgb):
    rgb_array = np.array([rgb])
    scaled_rgb = scaler.transform(rgb_array)
    predicted_label = svm_model.predict(scaled_rgb)
    return predicted_label[0]

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files.getlist('files')
    if len(files) != 6:
        return jsonify({'error': 'Exactly 6 images are required'}), 400

    final = []
    string_to_char = {
        "white": 'w',
        "green": 'g',
        "red": 'r',
        "blue": 'b',
        "yellow": 'y',
        "orange": 'o'
    }

    for i, file in enumerate(files):
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            im = Image.open(filepath).convert('RGB')
            centers = get_center_coordinates(im.width, im.height)
            face_edge_distance = min(im.width, im.height) // 7
            colors = [classify_color(average_rgb_in_circle(im, x, y, face_edge_distance)) for x, y in centers]

            # Adjust this part for actual user verification if needed
            ans = "".join([string_to_char[color] for color in colors])
            final.append(ans)
        else:
            return jsonify({'error': f'File {file.filename} is not allowed'}), 400

    with open("cube_data.txt", "w") as file:
        for element in final:
            file.write(element + "\n")

    # solver_path = os.path.join(os.path.dirname(__file__), 'Solver.exe')
    result = subprocess.run(["../Solver.exe", "cube_data.txt"], capture_output=True, text=True)
    
    if result.returncode == 0:
        response = {'result': result.stdout}
        print("C++ solver output:", result.stdout)
    else:
        response = {'error': result.stderr}
        print("C++ solver error:", result.stderr)
    print(jsonify(response))
    return jsonify({'result':response})

if __name__ == '__main__':
    app.run(debug=True)
