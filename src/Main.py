
#@title Imports and Install
#!pip install ultralytics

import os
import json
import torch
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Paths for JSON files and directories
data_dir = 'C:\\Work\\AIML\\DataSets\\BDD_DataSet\\data_dir\\data_dir'
#active_ds = 'BDD_50'
active_ds = 'BDD100K'
#active_ds = 'BDD10K'

train_Img_dir = os.path.join(data_dir, active_ds, 'train', 'images')
train_Label_dir = os.path.join(data_dir, active_ds, 'train', 'labels')
train_labels_json = os.path.join(data_dir, 'det_train.json')

val_Img_dir = os.path.join(data_dir, active_ds, 'val', 'images')
val_Label_dir = os.path.join(data_dir, active_ds, 'val', 'labels')
val_labels_json = os.path.join(data_dir, 'det_val.json')


#@title YOLO v11
# Load the pretrained YOLO model
model_path = train_Img_dir = os.path.join(data_dir, 'runs\\detect\\train7\\weights\\best.pt')  # Path to the locally saved pretrained model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The pretrained model file '{model_path}' does not exist.")

# Load the model
model = YOLO(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model is training on : {device}")
config_data_dir = 'C:\\Work\\AIML\\DataSets\\BDD_DataSet\\data_dir\\data_dir\\local_config.yaml'

# Train the model
train_results = model.train(
    data=config_data_dir,  # path to dataset YAML
    epochs=5,  # number of training epochs
    #imgsz=640,  # training image size
    pretrained=True,
    device=device,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate the model on the validation set
eval_results = model.val()
print("Evaluation Results:", eval_results)
