To install the data, you can run the install_data.sh script. Alternatively, you can install them manually 
by following the instructions on [Visual Genome Python Driver](https://github.com/ranjaykrishna/visual_genome_python_driver) and 
[COCO API WRAPPER](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools). The wrappers and api are already installed
in the environment. 

To format the Visual Genome object, run the notebook.

## Install APIs
Please replace the empty directories `pycocotools` and `visual_genome_python_driver` with
the one with the same name in 
[COCO Python API](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) 
and [Visual Genome Python Driver](https://github.com/ranjaykrishna/visual_genome_python_driver).

## Setup downloaded datasets
### COCO
Please download COCO2017 and annotations to `coco`:
```commandline
coco/
    ├── annotations/
    │   ├── instances_train2017.json
    │   ├── instances_val2017.json
    │   ├── ... 
    ├── train2017/
    ├── val2017/
    └── ...
```

### Visual Genome
Please download the images to `vg`:
```commandline
vg/
    ├── VG_100k/
    │   ├── 1.jpg
    │   ├── ... 
    └── ...
```
- Preprocessed annotations can be directly used to train explainers is also provided as 
`vg/vg_labels.pkl` and `vg/vg_objects.json`.