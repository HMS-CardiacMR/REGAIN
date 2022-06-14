# Resolution-Enhancement-Generative-Adversarial-Inline-Network (REGAIN)

The repository contains a source code for Resolution-Enhancement-Generative-Adversarial-Inline-Network (REGAIN). REGAIN is a generative adversarial neural network for REGAINing image sharpness and Spatial resolution. The trained network generates a resolution-enhanced image in the Phase-encoding (ky) direction in MRI.

## Requirements

For training and testing: pytorch, numpy
For GUI-based testing: pytorch, numpy, scipy, PyQt5, pydicom,  PIL, qimage2ndarray, cv2



## File Descriptions
-Config.py: The configuratioFile Descriptionsn file for defining a cuda machine, dataroot folder, and hyperparameters during training

-Dataset.py: Specefies the training dataset loading and data augumentation

-Models.py: Implementation of REGAIN

-Train.py: The numpy magnitude images are loaded and the training is performed. (This is what you run for training)

-Test.py: The numpy magnitude images are loaded and the testing is performed. (This is what you run for testing)

-GUI-Testing.py: The DICOM files are loaded and visualize the result of REGAIN (This is what you run for off-line Reconstruction)

