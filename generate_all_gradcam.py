import skimage.io
import skimage.measure
import numpy as np
import pathlib
import lib_aug_gradcam as auggc


model_name = "vgg16"
do_crop_resize = True
batch_size = 1000
top_n = 5
num_aug = 100
data_path = pathlib.Path("")  # TODO: paste here the path to the directory of the competition (http://image-net.org/challenges/LSVRC/2015/index#resources)
all_files = sorted(data_path.glob("*.JPEG"))

dict_vgg = auggc.get_vgg_dict()
dict_comp = auggc.get_competition_dict()

# model initialization
gradcam_model = auggc.GradCamModel(model_name=model_name)
augmenter = auggc.Augmenter(num_aug=num_aug)
cnt_batch = -1
batch_cams = []
batch_preds = []
batch_probs = []
batch_angles = []
batch_shift = []
batch_crop = []
batch_orig_imsz = []
for ii, filepath in enumerate(all_files):

    print(ii)
    if ii > 100000:
        break

    img = skimage.io.imread(str(filepath))

    orig_imsz = img.shape[0:2]
    if do_crop_resize:
        img, crop = auggc.crop_and_resize_img(img)
    else:
        img = auggc.resize_img(img)
        crop = (0, 0, 0)

    angles = np.random.uniform(-0.5, 0.5, num_aug)
    angles[0] = 0
    shift = np.random.uniform(-30, 30, (num_aug, 2))
    shift[0] = np.array([0, 0])

    img_batch = augmenter.direct_augment(img, angles, shift)
    top_k_cams, top_k_preds, top_k_probs = gradcam_model.compute_top_k_cams(img_batch)

    batch_cams.append(top_k_cams)
    batch_preds.append(top_k_preds)
    batch_probs.append(top_k_probs)
    batch_angles.append(angles)
    batch_shift.append(shift)
    batch_crop.append(crop)
    batch_orig_imsz.append(orig_imsz)

    if np.mod(ii+1, batch_size) == 0:
        cnt_batch = cnt_batch + 1
        filename = "data/cams/batch_{}".format(cnt_batch)
        np.savez(filename,
                 cams=batch_cams,
                 preds=batch_preds,
                 probs=batch_probs,
                 angles=batch_angles,
                 shift=batch_shift,
                 crop=batch_crop,
                 imsz=batch_orig_imsz)

        batch_cams = []
        batch_preds = []
        batch_probs = []
        batch_angles = []
        batch_shift = []
        batch_crop = []
        batch_orig_imsz = []
