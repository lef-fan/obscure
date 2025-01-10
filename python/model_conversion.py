import os
import shutil
from ultralytics import YOLO


model_version = "yolov8l-seg.pt"
input_shape = (288,480)
converted_model_name = model_version.split(".")[0] + ".torchscript"
converted_model_new_name = model_version.split(".")[0] + "_" + str(input_shape[0]) + "-" + str(input_shape[1]) + ".torchscript"
converted_model_path = "models"

model = YOLO(model_version)
model.export(format="torchscript", imgsz=input_shape)

if os.path.exists(model_version):
    os.remove(model_version)
    print(f"Removed: {model_version}")
else:
    print(f"Model not found: {model_version}")

if os.path.exists(converted_model_name):
    shutil.move(converted_model_name, os.path.join(converted_model_path, converted_model_new_name))
    print(f"Moved: {converted_model_name} to {os.path.join(converted_model_path, converted_model_new_name)}")
else:
    print(f"File not found: {converted_model_name}")