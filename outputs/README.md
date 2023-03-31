# FACT-AI-group18
This is an attempt to reproduce the results and findings of [LaViSE](https://github.com/YuYang0901/LaViSE)

# Requirements
## Environment
The environment necessary to run the code can be created with the environment_lisa.yml file when running on lisa cluster and the environment_windesk when running locally on windows. 

## Data
To install the data, you can run the install_data.sh script. Alternatively, you can install them manually 
by following the instructions on [Visual Genome Python Driver](https://github.com/ranjaykrishna/visual_genome_python_driver) and 
[COCO API WRAPPER](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools). The wrappers and api are already installed
in the environment. 

To format the Visual Genome object, run the notebook.

# Running the model
There are several job scripts to run the code. 

Alternatively, you can manually run
'''
python train_explainer.py --refer <reference_dataset> --epochs <number_of_epochs> --name <run_name>
'''
on your desktop to train the explainer, and
'''
python infer_filter.py --refer <reference_dataset> --name <run_name> 
'''
to visualize the filters. 

# Original README of LaViSE page
This is the official repository for paper "Explaining Deep Convolutional Neural Networks via Unsupervised 
Visual-Semantic Filter Attention" to appear in CVPR 2022. 

Authors: Yu Yang, Seungbae Kim, Jungseock Joo

[//]: # (## Requirements)

## Datasets
### [Common Objects in Context (COCO)](https://cocodataset.org/#home) 

- Please follow the instructions in the 
[COCO API README](https://github.com/cocodataset/cocoapi) and 
[here](data/README.md) to download and setup the COCO data.

### [Visual Genome (VG)](https://visualgenome.org/)

- Please follow the instructions in the 
[README of the python wrapper for the Visual Genome API](https://github.com/ranjaykrishna/visual_genome_python_driver) 
and [here](data/README.md).

### [GloVe](https://nlp.stanford.edu/projects/glove/)

- We load the pretrained GloVe word embeddings directly from the 
[torchtext](https://torchtext.readthedocs.io/en/latest/vocab.html#glove) library.

### Social Media <u>P</u>hotographs <u>o</u>f US <u>P</u>oliticians (PoP)

- The list of entities used to discover new concepts is provided in `data/entities.txt`.

## Getting started 

### Requirements

Required packages can be found in `requirements.txt`.

### Usage

Train an explainer with

```commandline
python train_explainer.py
```

Explain a target filter of any model with

```commandline
python infer_filter.py
```

More features will be added soon! 🍻

## Citation
```
@inproceedings{yang2022explaining,
    author    = {Yang, Yu and Kim, Seungbae and Joo, Jungseock},
    title     = {Explaining Deep Convolutional Neural Networks via Unsupervised Visual-Semantic Filter Attention},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2022},
}
```
