from flask import Flask, request, jsonify,send_from_directory
#import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__,static_folder='static',template_folder='templates')

# Load the trained model
model = load_model('mnist_cnn_model.h5')
print("model",model)

def preprocess_image(image_data):
    print("Image is preprocessed")
    # Decode base64 image
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))

    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28
    image = image.resize((28, 28))
    # Normalize pixel values
    image_array = np.array(image) / 255.0

    #image_array = 1 - image_array
    # Reshape for model input
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array



@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/predict-digit', methods=['POST'])
def predict_digit():
    data = request.get_json()
    image_data = data['image']
    image_array = preprocess_image(image_data)
    print("Image pixel values:", image_array[0, :, :, 0])
    print("Preprocessed Image Shape:", image_array.shape)  # Check the shape
    print("Preprocessed Image Data:", image_array) 
    # Plot the image (drop the extra dimension for visualization)

    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions[0])
    return jsonify({'prediction': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
