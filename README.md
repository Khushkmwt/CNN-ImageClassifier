
# 🖼️ CNN Image Classification

## 🚀 Overview
This project implements a **Convolutional Neural Network (CNN)** using TensorFlow and Keras to classify images of **cats and dogs**. The model is trained on a dataset of labeled images and can predict whether a given image contains a cat or a dog.

## 📂 Dataset
The dataset consists of three main directories:

📌 **dataset/training_set/** - Contains training images categorized into subdirectories (`cats/` and `dogs/`).

📌 **dataset/test_set/** - Contains test images for model validation.

📌 **dataset/single_prediction/** - Contains a single image for prediction.

## 🔧 Requirements
Make sure you have the necessary dependencies installed:

```bash
pip install tensorflow numpy
```

## 🏗️ Model Architecture
1. 🎛️ **Convolutional Layers** - Extract features from images using 2D convolution.
2. 📉 **Max Pooling** - Reduces the spatial dimensions.
3. 🔄 **Flattening** - Converts the matrix into a vector.
4. 🔗 **Fully Connected Layers** - Dense layers for classification.
5. 🎯 **Output Layer** - Uses a **sigmoid** activation function to classify images into two categories (**cat/dog**).

## 🏋️‍♂️ Training the Model
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 📊 Data Augmentation for training images
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# 📂 Loading the dataset
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# 🏗️ Building the CNN
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# ⚙️ Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🎯 Training the CNN
cnn.fit(x=training_set, validation_data=test_set, epochs=25)
```

## 🔍 Making Predictions
```python
import numpy as np
from tensorflow.keras.preprocessing import image

# 📷 Load and preprocess the image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# 🤖 Predict the class
result = cnn.predict(test_image)

if result[0][0] == 1:
    prediction = '🐶 Dog'
else:
    prediction = '🐱 Cat'
print(f'Prediction: {prediction}')
```

## 📊 Results
✔️ The model successfully classifies images as either **cats** 🐱 or **dogs** 🐶 with high accuracy!

## 🎯 Conclusion
This project demonstrates the power of **CNNs** in **image classification tasks**. You can improve accuracy by:
- 📈 Increasing dataset size
- 🛠️ Tweaking hyperparameters
- 🔄 Using pre-trained models like **VGG16**

## ✨ Author
Developed by **Dilkhush Kumawat** 🚀

---
📌 *Feel free to modify and experiment with the model for better performance!*

