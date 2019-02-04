# Segmentation of nuclei using Ilastik

By Molnar Csaba @csmolnar

A random forrest classifier is trained after loading the training images. Some notes of how to run this project:

Label images are transformed into sparse pixel samples. Completely random.
Sampling rates: 10% background - 10% foreground - 50% boundaries.
The sampling is run in Matlab. Run for all training images. This produces the following gray scale values: 
000000 0,0,0        Nothing
FFFFFF 255,255,255.    Boundaries
AAAAAA 170,170,170.    Background
616161 97,97,97.    Foreground

1) Create a new project in Ilastik and load the training data (only original images).
   By default, it uses Random Forests (more details in the paper).

2) Features selected for training.
   Screenshot in Slack.

3) Create classes in the project.
   There is a skeleton project file with the basic Ilastik configuration.

4) Run the Python script to generate a new Ilastik project file with the desired labeled images.
   Parameters: 1- skeleton project file.
               2- folder of labels
   This script has dependencies with Ilastik packages
   This generates the new project that can be used for training.
   This is a bigger file that can be reproduced following the steps before.

5) Open and run the new project file in Ilastik
   Start training with Live Update.

6) When done, we can select what to export: 
   Choose probability maps.
   Convert to Data Type: unsigned 16-bits
   File format: PNG

7) Load test images: use batch processing for opening the test images.
   Process all files: generates the probability maps for test images.

8) Postprocessing: use the Jupyter Notebook to transform probability maps into label matrices
   The notebook is preconfigured with the right parameters.
   Generates outputs in a new folder.

9) Evaluation: object dilation = 3
   

