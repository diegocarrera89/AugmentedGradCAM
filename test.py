from matplotlib import pyplot as plt
import numpy as np
import skimage.measure

import lib_aug_gradcam as auggc

# test image name
img_name = "cat_dog.png"
idx = 242  # index for the true dog true class

# load the test image
img_path = "./imgs/{}".format(img_name)
img = skimage.io.imread(img_path)

# super resolution parameters
num_aug = 100
learning_rate = 0.001
lambda_eng = 0.0001 * num_aug
lambda_tv = 0.002 * num_aug
thresh_rel = 0.15
num_iter = 200

# augmentation parameters
angle_min = -0.5  # in radians
angle_max = 0.5
angles = np.random.uniform(angle_min, angle_max, num_aug)
shift_min = -30
shift_max = 30
shift = np.random.uniform(shift_min, shift_max, (num_aug, 2))
# first Grad-CAM is not augmented
angles[0] = 0
shift[0] = np.array([0, 0])

augmenter = auggc.Augmenter(num_aug, (224, 224))
superreso = auggc.Superresolution(augmenter=augmenter, learning_rate=learning_rate, camsz=[14,14])


gradcam_model = auggc.GradCamModel(model_name="vgg16")
img_batch = augmenter.direct_augment(img, angles, shift)
idx_batch = [idx for ii in range(num_aug)]

# compute `num_aug` Grad-CAMs
cams = gradcam_model.compute_cam(img_batch, idx_batch)[:, :, :, np.newaxis]

# compute the four approaches to Grad-CAMs
cam_full_single = superreso.super_single(cams).squeeze()
cam_full_max = superreso.super_max(cams, angles, shift).squeeze()
cam_full_avg = superreso.super_avg(cams, angles, shift).squeeze()
cam_full_tv = superreso.super_mixed(cams / np.max(cams, axis=(1, 2, 3), keepdims=True),
                                    angles, shift, lmbda_tv=lambda_tv, lmbda_eng=lambda_eng, niter=num_iter).squeeze()

# show the results
plt.figure()

# single Grad-CAM
plt.subplot(2, 2, 1)
plt.imshow(cam_full_single, cmap="jet")
plt.title("Single")

# max aggregation
plt.subplot(2, 2, 2)
plt.imshow(cam_full_max, cmap="jet")
plt.title("Max")

# average aggregation
plt.subplot(2, 2, 3)
plt.imshow(cam_full_avg, cmap="jet")
plt.title("Avg")

# super resolution aggregation
plt.subplot(2, 2, 4)
plt.imshow(cam_full_tv, cmap='jet')
plt.title("Augmented")

plt.show()

# overlay CAMs to input image
superreso.overlay(cam_full_single, img, name="single")
superreso.overlay(cam_full_max, img, name="max")
superreso.overlay(cam_full_avg, img, name="avg")
superreso.overlay(cam_full_tv, img, name="augmented")

# compute bounding boxes
bbox_single = auggc.get_bbox_from_cam(cam_full_single)
bbox_max = auggc.get_bbox_from_cam(cam_full_max)
bbox_avg = auggc.get_bbox_from_cam(cam_full_avg)
bbox_tv = auggc.get_bbox_from_cam(cam_full_tv)

# plot boxes
rect_single = auggc.get_rectangle(bbox_single, color="red")
rect_max = auggc.get_rectangle(bbox_max, color="blue")
rect_avg = auggc.get_rectangle(bbox_avg, color="magenta")
rect_tv = auggc.get_rectangle(bbox_tv, color="green")

fig, ax = plt.subplots(1, 1)
ax.imshow(img)
ax.add_patch(rect_single)
ax.add_patch(rect_max)
ax.add_patch(rect_avg)
ax.add_patch(rect_tv)
ax.legend((rect_single, rect_max, rect_avg, rect_tv), ("Single", "Max", "Avg", "Augmented"))
plt.show()



