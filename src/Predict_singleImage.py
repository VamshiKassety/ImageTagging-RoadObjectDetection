import os
import json
import torch
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# Function to Render Predictions
def render_prediction(image_path, model):
    """
    Take an image as input and render the prediction from the trained model.

    Args:
        image_path (str): Path to the input image.

    Returns:
        None: Displays the image with predictions overlaid.
    """
    # Load the image
    img = Image.open(image_path)
    
    # Run the model on the image
    results = model.predict(source=image_path, device=device, show=False)
    
    # Render the results on the image
    results_image = results[0].plot()  # Plot predictions on the image
    
    # Display the image with predictions
    plt.figure(figsize=(10, 10))
    plt.imshow(results_image)
    plt.axis("off")
    plt.show()

# Example Usage
data_dir = 'C:\\Work\\AIML\\DataSets\\BDD_DataSet\\data_dir\\data_dir\\'
#MODEL_PATH = os.path.join(data_dir, 'runs\\detect\\train7\\weights\\best.pt')
test_image = os.path.join(data_dir, '\\predict_videos\\img2.png')
model = os.path.join(data_dir, 'runs\\detect\\train7\\weights\\best.pt')
render_prediction(test_image, model)