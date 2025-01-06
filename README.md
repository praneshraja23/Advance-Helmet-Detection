# Motorcycle Helmet Detection using YOLOv8

## Overview

This project implements a **Motorcycle Helmet Detection** system using the state-of-the-art **YOLOv8** (You Only Look Once) object detection algorithm. The primary goal of this project is to improve safety by detecting whether a motorcyclist is wearing a helmet or not. The system utilizes a pre-trained YOLOv8 model, fine-tuned on a custom dataset containing images of motorcyclists, to accurately classify images based on whether the subject is wearing a helmet.

The project leverages **Roboflow** for easy dataset management, **YOLOv8** for the detection model, and **Google Colab** for executing the training and inference processes. The resulting model can be used to automatically assess helmet-wearing compliance, a crucial factor in road safety enforcement.

---

## Features

- **Helmet Detection**: Detects whether a motorcyclist is wearing a helmet or not.
- **Custom Dataset**: Trained on a custom dataset obtained from Roboflow with images labeled for helmet detection.
- **Model Training**: Trains the YOLOv8 model on the custom dataset for 25 epochs.
- **Validation & Testing**: Validates the trained model on a test set, with performance metrics like confusion matrices displayed.
- **Inference**: Provides real-time predictions with the trained model, including saving the output predictions.
- **Roboflow Integration**: Leverages Roboflow's easy-to-use platform to manage and download datasets for training and testing.

---

## Requirements

To run this project, ensure you have the following dependencies installed:

- **Python** >= 3.6
- **Ultralytics YOLOv8**: For object detection using the YOLO architecture.
- **Roboflow**: For dataset management and model handling.
- **pip**: To install the required Python packages.

You can install the necessary dependencies using the following commands:

```bash
pip install ultralytics==8.2.103
pip install roboflow
```
---
## Dataset

The dataset used in this project is hosted on **Roboflow**. It consists of images of motorcyclists, labeled for helmet detection. The dataset was downloaded in **YOLOv8 format** and contains both training and validation data.

You can access the dataset on Roboflow via the following link:

[Motorcycle Helmet Detection Dataset](https://roboflow.com)

---
## Training the Model

The model is trained using the following command:

```bash
yolo task=detect mode=train model=yolov8s.pt data={dataset_location}/data.yaml epochs=25 imgsz=800 plots=True
```
This command trains the YOLOv8 model using a base model yolov8s.pt for 25 epochs with an image size of 800 pixels.

---
## Model Evaluation

After training, the model is evaluated using a validation set with the following command:

```bash
yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset_location}/data.yaml
```

This section will help explain how to evaluate the model's performance in the README file. Be sure to replace `{HOME}` and `{dataset_location}` with the actual paths where your model and dataset are stored.

---
## Making Predictions

Once trained, the model can be used to predict whether a motorcyclist is wearing a helmet on unseen images. To perform inference on test images, use the following command:

```bash
yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset_location}/test/images save=True
```

This section can be added under the **Making Predictions** heading in your README to explain how to use the trained model to make predictions on new test images.

---
## Results

After running the predictions, the following results can be expected:

- **Confusion Matrix**: Provides a visualization of the classification performance, highlighting true positives, false positives, true negatives, and false negatives.
  
- **Predicted Images**: Displays test images with bounding boxes around detected helmets, showing the model's ability to identify whether a motorcyclist is wearing a helmet.

- **Accuracy**: The model achieves high accuracy in helmet detection, ensuring reliable performance for real-time safety compliance assessments.

