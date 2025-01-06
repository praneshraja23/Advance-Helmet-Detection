# Importing necessary library to interact with the operating system
import os

# Getting the current working directory (path where the script is running)
HOME = os.getcwd()
print(HOME)

# Install the ultralytics package, version 8.2.103
!pip install ultralytics==8.2.103 -q

# Clear the output of the previous cells (helpful for Jupyter Notebooks)
from IPython import display
display.clear_output()

# Check for system dependencies and configurations for ultralytics
import ultralytics
ultralytics.checks()

# Import YOLO class from ultralytics
from ultralytics import YOLO

# Import display and image functionality to display results
from IPython.display import display, Image

# Change the working directory back to the original home directory
%cd {HOME}

# Install the roboflow package
!pip install roboflow

# Initialize the Roboflow API with your API key
from roboflow import Roboflow
rf = Roboflow(api_key="3N6Pnr6DqUJh82oqEP3A")

# Access the specific project in your Roboflow workspace
project = rf.workspace("innovatech").project("motorcycle-helmet-q0wmd")
version = project.version(1)
dataset = version.download("yolov8")

# Change directory back to the home directory
%cd {HOME}

# Train a YOLOv8 model on the dataset for 25 epochs
!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=25 imgsz=800 plots=True

# List the contents of the training output directory
!ls {HOME}/runs/detect/train/

# Display confusion matrix image from the training results
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=600)

# Change directory back to home
%cd {HOME}

# Validate the trained model using the best weights obtained from training
!yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml

# Change directory back to home
%cd {HOME}

# Run predictions on the test images and save the results
!yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True

# Import glob for file manipulation and display images
import glob
from IPython.display import Image, display

# Define the base path where the folders are located
base_path = '/content/runs/detect/'

# List all directories that start with 'predict' in the base path
subfolders = [os.path.join(base_path, d) for d in os.listdir(base_path)
              if os.path.isdir(os.path.join(base_path, d)) and d.startswith('predict')]

# Find the latest folder by modification time
latest_folder = max(subfolders, key=os.path.getmtime)

# Get a list of image paths from the latest folder and display a range of them
image_paths = glob.glob(f'{latest_folder}/*.jpg')[3:10]

# Display each image
for image_path in image_paths:
    display(Image(filename=image_path, width=600))
    print("\n")
