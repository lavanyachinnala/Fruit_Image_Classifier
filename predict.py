import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("fruit_classifier.h5")

# Change this to the image you want to test
img_path = "sample.jpg"

img = image.load_img(img_path, target_size=(100,100))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)

# Class names in alphabetical order
classes = [
    "apple",
    "banana",
    "mango",
    "orange",
    "papaya",
    "strawberry",
    "watermelon"
]

predicted_class = classes[np.argmax(prediction)]

print("Predicted Fruit:", predicted_class)