# Augmented Grad-CAM
![Single Grad-CAM](imgs/single.png?raw=true "Single Grad-CAM")
![Augmented Grad-CAM](imgs/augmented.png?raw=true "Augmented Grad-CAM")

Python code to reproduce the experiments of Augmented Grad-CAM.

## Configuration
To reproduce experiments on the [ImageNet validation set](http://image-net.org/challenges/LSVRC/2015/index#resources), download it and add its directory path in the `generate_all_gradcam.py` script. In order to satisfy code dependencies, we suggest to configure a virtual environment using `virtualenv` and install the required modules, specified in `config/requirements.txt`.

First, create the virtual environment in the root directory
```
virtualenv -p python3 camTfEnv
```
Your directory tree should now look as follows

```
.
├── camTfEnv
├── config
    └── requirements.txt
├── data
    └── cams
├── dicts
│   ├── map_clsloc.txt
│   └── synset.txt
├── generate_all_gradcam.py
├── generate_submission_files.py
├── imgs
│   ├── augmented.png
│   ├── avg.png
│   ├── cat_dog.png
│   ├── max.png
│   └── single.png
├── lib_aug_gradcam.py
└── test.py
```

Then, activate the new virtual environment
```
source camTfEnv/bin/activate
```

And install the required modules
```
pip install -r config/requirements.txt
```

## Evaluation: Weak Localization
Finally, we compute the Grad-CAMs over the images contained in `data`, using
```
python generate_all_gradcam.py
```

Grad-CAMs are stored in batches as `.npz` files inside `data/cams`. 
Then, we compute the bounding boxes for weak localization over the generated Grad-CAMs using
```
python generate_submission_files.py
```

This generates a `.txt` file the directory root reporting for each image in `data`, the corresponding classification and the vertices of the corresponding bounding box.

## Examples
To reproduce the images in the heading of this repository, activate the virtual environment and launch the following script
```
source camTfEnv/bin/activate
python test.py
```

 ### Notes
 In the ImageNet experiments scripts, we decouple Grad-CAMs computation and aggregation (via `tensorflow`) due to the large size of the validation set. Moreover, we process images in batches to avoid memory saturation.
