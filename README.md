# KCPose


## Requirements
- Python 3.9
- numpy
- sklearn
- scipy
- matplotlib
- pymesh2
- CUDA version: 11.4 (depends on YOLO model version you use)


## Usage
### Key Component Extraction Module
Extract key components via following intruction:


    python $root/src/key_component_extraction/main.py $cad_model_path


After execution, the location, normal vector, class name of each key component will be saved in `$root/data/key_component`.


### Training on YCB-Object set
1. You have to download the [The YCB-Video 3D Models](https://rse-lab.cs.washington.edu/projects/posecnn/) first.
2. Generate training data for [Scaled-YOLOv4](https://github.com/AlexeyAB/darknet):
    - Add target environment's background images into `data/background`.
    - Create a blender file in `$root/src/blender/`.
    - Import your target CAD model and set the CAD model's name in Blender.
    - Load the code `main.py` and `util.py` into Blender, where the code is saved in  `path: $root/src/blender`
    - Set parameters in `/src/blender/settings.json`.
    - Run `main.py` in Blender to generate training data: images and xml annotations.
    - Run `xml_to_txt.py` to transform <b>xml annotations</b> to <b>yolo annotations</b>.
    - Follow the instruction of [Scaled-YOLOv4](https://github.com/AlexeyAB/darknet) to train the key component detection model.


### Evaluation
- After key components's bounding box acquired, you can use tools in `path: $root/src/pose` to evaluate pose.


### Evaluation
Checkpoint of example object and demo YCB Object is saved in [here](https://drive.google.com/drive/folders/1TrWUa66O2xOMGDqyqs5PQOSup-U6rJjV).
