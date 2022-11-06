#Dense extreme inception network for edge detection

#edge detection with cnn
"""1. HED (2015), 2. RCF (2017), 3. BDCN (2019), 4. CATS(2021), 5. dexined (2021)"""

"""Dexined - it doesnot use, in the training stage,
pretrained weights
- it has less hyperparameter tunning
- BIPED (Dataset)
- Model have 35Million parameters model - quite similar image classification and recognition"""

"""Try dexined: 
1. Github: https://github.com/xavysp/DexiNed
2. Paper: https://arxiv.org/abs/2112.02250"""

"""
STEPS:
1. Download the github
2. And download latest pytorch version dexined checkpoints through github
3. save the model in this path to dexined folder: Checkpoints/BIPED/10/
4. run - python main.py
5. If you custom train change the main.py file and change the dataset
"""
