# Augmented Grad-CAM
![Single Grad-CAM](imgs/single.png?raw=true "Single Grad-CAM")
![Augmented Grad-CAM](imgs/augmented.png?raw=true "Augmented Grad-CAM")

Python code to reproduce the experiments of Augmented Grad-CAM.

## Examples
To reproduce the above images, simply launch the following command within the cloned repository
```
python test.py
```

## Usage
To reproduce experiments on the [ImageNet validation set](http://image-net.org/challenges/LSVRC/2015/index#resources), download it and add its directory path in `generate_all_gradcam.py` script. Make sure your directory tree looks as follows
```
.
├── data
│   ├── cams_resnet50
│   └── cams_vgg16
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
Then, launch
```
python generate_all_gradcam.py
python generate_submission_files.py
```

 ### Notes
 In the ImageNet experiments scripts, we decouple Grad-CAMs generation and aggregation due to the large size of the validation set. Moreover, we process images in batches to avoid memory saturation.
