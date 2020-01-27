import numpy as np
import pathlib
import lib_aug_gradcam as auggc


model_name = "vgg16"  # or "resnet50"
do_crop_resize = True
data_path = pathlib.Path("data/cam_" + model_name)

# super resolution parameters
num_aug = 100
lambda_eng = 0.0001 * num_aug
lambda_tv = 0.002 * num_aug
thresh_rel = 0.15
num_iter = 200

nbatch = 50
nimg = 1000
top_n = 5

# model initialization
augmenter = auggc.Augmenter(num_aug)
superreso = auggc.Superresolution(augmenter)

# submission files initialization
subfile_single = open("submissions/sub_single.txt", 'w')
subfile_max = open("submissions/sub_max.txt", 'w')
subfile_avg = open("submissions/sub_avg.txt", 'w')
subfile_tv = open("submissions/sub_tv.txt", 'w')

for i_batch in range(nbatch):
    batch = np.load("data/cams_{0}/batch_{1:d}.npz".format(model_name, i_batch))
    batch_cams = batch["cams"]
    batch_preds = batch["preds"]
    batch_probs = batch["probs"]
    batch_angles = batch["angles"]
    batch_shift = batch["shift"]
    batch_crop = batch["crop"]
    batch_orig_imsz = batch["imsz"]

    for idx_img in range(nimg):
        print("Batch: {0:d}, img: {1:d}".format(i_batch+1, idx_img+1))

        top_k_cams = batch_cams[idx_img, 0:num_aug, :, :, :]
        top_k_preds = batch_preds[idx_img, 0, :]
        angles = batch_angles[idx_img, 0:num_aug]
        shift = batch_shift[idx_img, 0:num_aug]
        crop = batch_crop[idx_img]
        orig_imsz = batch_orig_imsz[idx_img]

        all_cam_full_max = []
        all_cam_full_single = []
        all_cam_full_tv = []
        all_cam_full_avg = []
        for k in range(top_n):
            cams = top_k_cams[:, :, :, k][:, :, :, np.newaxis]

            cam_full_single = superreso.super_single(cams).squeeze()
            cam_full_max = superreso.super_max(cams, angles, shift).squeeze()
            cam_full_avg = superreso.super_avg(cams, angles, shift).squeeze()
            cam_full_tv = superreso.super_mixed(cams / np.max(cams, axis=(1, 2, 3), keepdims=True),
                                                angles,
                                                shift,
                                                lmbda_tv=lambda_tv,
                                                lmbda_eng=lambda_eng,
                                                niter=num_iter).squeeze()
            all_cam_full_single.append(auggc.pad_and_resize_cam(cam_full_single, orig_imsz, crop, order=1))
            all_cam_full_max.append(auggc.pad_and_resize_cam(cam_full_max, orig_imsz, crop, order=1))

            all_cam_full_avg.append(auggc.pad_and_resize_cam(cam_full_avg, orig_imsz, crop, order=1))
            all_cam_full_tv.append(auggc.pad_and_resize_cam(cam_full_tv, orig_imsz, crop, order=1))

        bbox_str_single = auggc.get_bbox_str(all_cam_full_single, top_k_preds, thresh_rel)
        bbox_str_max = auggc.get_bbox_str(all_cam_full_max, top_k_preds, thresh_rel)
        bbox_str_avg = auggc.get_bbox_str(all_cam_full_avg, top_k_preds, thresh_rel)
        bbox_str_tv = auggc.get_bbox_str(all_cam_full_tv, top_k_preds, thresh_rel)

        subfile_single.write(bbox_str_single + '\n')
        subfile_max.write(bbox_str_max + '\n')
        subfile_avg.write(bbox_str_avg + '\n')
        subfile_tv.write(bbox_str_tv + '\n')

subfile_single.close()
subfile_max.close()
subfile_avg.close()
subfile_tv.close()



