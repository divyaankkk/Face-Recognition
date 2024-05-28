# Face Recognition
 
his Python script facilitates the collection of images from a webcam and performs data augmentation for training purposes. It captures images from a webcam, saves them, and then applies various augmentation techniques to generate augmented images and corresponding annotations for training machine learning models, specifically a FaceTracker model in this case.

Features
Webcam Image Collection: Captures a specified number of images from the webcam and saves them to a specified directory.
Data Augmentation: Utilizes the Albumentations library to apply various augmentation techniques such as random cropping, horizontal and vertical flipping, brightness and contrast adjustments, gamma correction, and RGB shifting.
Image Annotation: Reads annotations in JSON format corresponding to the captured images and generates augmented annotations for the augmented images.
TensorFlow Data Pipeline: Loads the augmented images and annotations into TensorFlow data pipelines for training a machine learning model.
FaceTracker Model: Defines and trains a FaceTracker model using TensorFlow/Keras, which predicts the presence of a face in an image and provides bounding box coordinates around the detected face.

Dependencies
Python 3.x
OpenCV (cv2)
TensorFlow
Albumentations
Matplotlib