##Code to train the model MNIST
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical # type: ignore

#Loading the MNIST Datasets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(mnist.load_data())

# Reshape the images to match the input shape for CNN [samples, width, height, channels]
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Normalize pixel values to the range [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Convert labels to one-hot encoded vectors
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 2. Build the CNN model
model = models.Sequential()

# Add Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten the output and add fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 10 classes for digits 0-9

# 3. Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']) 
# 4. Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# 5. Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.4f}')

# 6. Save the trained model
model.save('mnist_cnn_model.h5')




# Load MNIST test dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data similarly as the input
x_test = x_test / 255.0
x_test = np.expand_dims(x_test, axis=-1)

# Test the model with a few examples
predictions = model.predict(x_test[:10])

# Print predicted labels and actual labels
predicted_labels = np.argmax(predictions, axis=1)
print("Predicted:", predicted_labels)
print("Actual:", y_test[:10])
