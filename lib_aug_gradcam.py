import cv2
import matplotlib.patches as patches
import numpy as np
import skimage.io
import skimage.transform
import skimage.measure
import tensorflow as tf
import tensorflow.keras as keras


sess = tf.InteractiveSession(graph=tf.Graph())


def get_competition_dict():
    file_path = "dicts/map_clsloc.txt"
    d = dict()
    with open(file_path, 'r') as file:
        for i_line, line in enumerate(file.readlines()):
            line = line.replace('\n', '')
            tokens = line.split(' ', 3)
            name = tokens[0]
            id = int(tokens[1])
            syn = tokens[2]
            d[name] = [id, syn]
    return d


def get_vgg_dict():
    file_path = "dicts/synset.txt"
    d = dict()
    with open(file_path, 'r') as file:
        for i_line, line in enumerate(file.readlines()):
            line = line.replace('\n', '')
            tokens = line.split(' ', 1)
            name = tokens[0]
            syn = tokens[1]
            d[i_line] = [name, syn]
    return d


def resize_img(img, target_imsz=(224, 224), order=1):
    img_new = img
    if len(img.shape) == 2:
        img_new = img_new[:, :, np.newaxis]
        img_new = np.concatenate((img_new, img_new, img_new), 2)
    img_new = skimage.transform.resize(img_new, target_imsz, order=order)
    return img_new * 255


def crop_and_resize_img(img, target_imsz=(224, 224)):
    """ Crop and resize `img` to match `target_imsz`.

    Images are cropped to be square images.

    :param img: the input image
    :param target_imsz: the target image size
    :return img: the resized image
    :return crop: the applied crop
    """
    img_new = img
    if len(img.shape) == 2:
        img_new = img_new[:, :, np.newaxis]
        img_new = np.concatenate((img_new, img_new, img_new), 2)

    n, m = img.shape[:2]

    if n < m:
        mt = int(np.floor((m-n)/2))
        mb = m - n - mt
        img_new = img_new[:, mt:-mb, :]
        crop = (1, mt, mb)
    elif n > m:
        nt = int(np.floor((n - m) / 2))
        nb = n - m - nt
        img_new = img_new[nt:-nb, :, :]

        crop = (0, nt, nb)
    else:
        crop = (0, 0, 0)
    img_new = resize_img(img_new, target_imsz)
    return img_new, crop


def resize_cam(cam, target_imsz, order):
    return skimage.transform.resize(cam, target_imsz, order=order)


def pad_and_resize_cam(cam, target_imsz, crop, order, mode="zero"):
    imsz = np.zeros(2)
    if crop[0] == 0:
        imsz[0] = target_imsz[0] - np.sum(crop[1:])
        imsz[1] = target_imsz[1]
    else:
        imsz[0] = target_imsz[0]
        imsz[1] = target_imsz[1] - np.sum(crop[1:])

    cam_new = resize_cam(cam, imsz, order=order)
    if np.sum(crop) == 0:
        cam_full = cam_new
    else:
        if crop[0] == 0:
            if mode == "zero":
                row_top = np.zeros((crop[1], cam_new.shape[1]))
                row_bottom = np.zeros((crop[1], cam_new.shape[1]))
            else:
                row_top = cam_new[0, :][np.newaxis, :]
                row_top = np.repeat(row_top, crop[1], 0)
                row_bottom = cam_new[-1, :][np.newaxis, :]
                row_bottom = np.repeat(row_bottom, crop[2], 0)
            cam_full = np.concatenate((row_top, cam_new, row_bottom), 0)
        else:
            if mode == "zero":
                col_top = np.zeros((cam_new.shape[0], crop[1]))
                col_bottom = np.zeros((cam_new.shape[0], crop[2]))
            else:
                col_top = cam_new[:, 0][:, np.newaxis]
                col_top = np.repeat(col_top, crop[1], 1)
                col_bottom = cam_new[:, -1][:, np.newaxis]
                col_bottom = np.repeat(col_bottom, crop[2], 1)
            cam_full = np.concatenate((col_top, cam_new, col_bottom), 1)
    return cam_full


def get_rectangle(bbox, color='r'):
    """ Get a rectangle patch from `bbox` for plotting purposes.

    :param bbox: input bounding box
    :param color: rectangle color
    :return rect: a colored rectangle patch
    """
    # Create a Rectangle patch
    (xmin, xmax, ymin, ymax) = bbox
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor="none")
    return rect


def order_bbox_from_sk(bbox):
    return bbox[1], bbox[3], bbox[0], bbox[2]


def get_bbox_from_gt(gt_root):
    """ Get the all the supervised bounding boxes (ground truth) having `gt_root` as root folder.

    :param gt_root: root folder for the ground truth bounding boxes
    :return all_objs: all supervised bounding boxes
    """
    all_objs = dict()

    all_objs_elem = gt_root.findall("./object")
    for obj_elem in all_objs_elem:
        id = obj_elem.find("./name").text

        if id not in all_objs.keys():
            all_objs[id] = []

        bbox_elem = obj_elem.find("./bndbox")

        xmin = int(bbox_elem.find("./xmin").text) - 1
        ymin = int(bbox_elem.find("./ymin").text) - 1
        xmax = int(bbox_elem.find("./xmax").text)
        ymax = int(bbox_elem.find("./ymax").text)

        all_objs[id].append((xmin, xmax, ymin, ymax))
    return all_objs


def get_bbox_from_cam(cam_full, thresh_rel=0.15):
    """ Compute the bounding box from `cam_full` according to `thresh_rel`.

    Fit a box around `cam_full` where its values are greater than the 15% of its maximum.

    :param cam_full: the input CAM
    :param thresh_rel: the threshold values (same as in https://arxiv.org/abs/1610.02391)
    :return: the bounding box for `cam_full`
    """
    thresh = thresh_rel * np.max(cam_full)
    mask = cam_full > thresh

    label_mask = skimage.measure.label(mask)
    all_regions = skimage.measure.regionprops(label_mask)

    all_areas = [region["area"] for region in all_regions]
    idx = np.argmax(all_areas)
    try:
        idx = idx[0]
    except:
        pass
    region = all_regions[idx]
    bbox = region.bbox

    return order_bbox_from_sk(bbox)

def bb_intersection_over_union(boxA, boxB):
    """ Compute the intersection over union (IOU) metrics between two bounding boxes.

    :param boxA: bounding box 1
    :param boxB: bounding box 2
    :return iou: the intersection over union of `boxA` and `boxB`
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[2], boxB[2])
    xB = min(boxA[1], boxB[1])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[1] - boxA[0] + 1) * (boxA[3] - boxA[2] + 1)
    boxBArea = (boxB[1] - boxB[0] + 1) * (boxB[3] - boxB[2] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


class GradCamModel:

    def __init__(self, top_k=5, model_name="vgg16"):
        graph = tf.get_default_graph()
        self.model_name = model_name.lower()
        if self.model_name == "vgg16":
            model = keras.applications.vgg16.VGG16()
            tns_full_prob = graph.get_tensor_by_name("predictions/Softmax:0")
            tns_lastfc = graph.get_tensor_by_name("predictions/BiasAdd:0")
            tns_conv = graph.get_tensor_by_name("block5_conv3/Relu:0")
            tns_input = graph.get_tensor_by_name("input_1:0")
        elif self.model_name == "resnet50":
            model = keras.applications.resnet50.ResNet50()
            tns_full_prob = graph.get_tensor_by_name("fc1000/Softmax:0")
            tns_lastfc = graph.get_tensor_by_name("fc1000/BiasAdd:0")
            tns_conv = graph.get_tensor_by_name("activation_48/Relu:0")
            tns_input = graph.get_tensor_by_name("input_1:0")
        else:
            model = None
            tns_full_prob = None
            tns_lastfc = None
            tns_conv = None
            tns_input = None
            ValueError("Model {} not supported".format(model_name))

        # input index for the target class
        tns_index = tf.placeholder("int32", [None])
        self.tns_index = tns_index

        model.build(tns_input)
        self.tns_full_prob = tns_full_prob
        self.tns_input = tns_input

        # CAM tensors for the input index
        tns_one_hot = tf.one_hot(tns_index, 1000, axis=-1)
        tns_loss = tf.reduce_sum(tf.multiply(tns_lastfc, tns_one_hot), axis=1)
        tns_grads = tf.gradients(tns_loss, tns_conv)[0]
        tns_coeff = tf.reduce_mean(tns_grads, axis=(1, 2), keepdims=True)
        tns_mult = tf.multiply(tns_coeff, tns_conv)
        tns_cam = tf.nn.relu(tf.reduce_sum(tns_mult, axis=3))
        self.tns_cam = tns_cam

        # top_k CAM tensors
        tns_top_k_prob, tns_top_k_pred = tf.math.top_k(tns_full_prob, k=top_k, sorted=True, name=None)
        all_tns_cam = []
        for k in range(top_k):
            tns_loss = tf.reduce_sum(tf.multiply(tns_lastfc, tf.one_hot(tns_top_k_pred[0][k], 1000)), axis=1)
            tns_grads = tf.gradients(tns_loss, tns_conv)[0]
            tns_coeff = tf.reduce_mean(tns_grads, axis=(1, 2), keep_dims=True)
            tns_cam = tf.nn.relu(tf.reduce_sum(tf.multiply(tns_coeff, tns_conv), axis=3, keepdims=True))
            all_tns_cam.append(tns_cam)
        tns_top_k_cam = tf.concat(all_tns_cam, axis=3)

        self.tns_top_k_prob = tns_top_k_prob
        self.tns_top_k_pred = tns_top_k_pred
        self.tns_top_k_cam = tns_top_k_cam

    def preprocess_input(self, image_batch):
        """ Pre-process `image_batch` to meet the specifications of the loaded model.

        :param image_batch: the input images
        :return all_img_pre: the pre-processed input images
        """
        if self.model_name == "vgg16":
            all_img_pre = [keras.applications.vgg16.preprocess_input(image.copy())[np.newaxis, :, :, :] for image in image_batch]
        elif self.model_name == "resnet50":
            all_img_pre = [keras.applications.vgg16.preprocess_input(image.copy())[np.newaxis, :, :, :] for image in image_batch]
        else:
            all_img_pre = None
        return all_img_pre

    def compute_full_prob(self, image_batch):
        """ Classify the images and return the probability values over the considered label space.

        :param image_batch: the input images
        :return full_prob: the probailities values for the predicted classes
        """
        all_img_pre = self.preprocess_input(image_batch)
        img_batch = np.concatenate(all_img_pre, 0)
        full_prob = sess.run(self.tns_full_prob, feed_dict={self.tns_input: img_batch})
        return full_prob

    def compute_cam(self, image_batch, index_batch):
        """ Compute the CAMs for `image_batch` w.r.t. to the class referenced by `index_batch`

        :param image_batch: the input images for the CAM algorithm
        :param index_batch: the indicies corresponding to the desired class
        :return cam: the CAMs
        """
        all_img_pre = self.preprocess_input(image_batch)
        if len(all_img_pre) == 1:
            img_batch = all_img_pre[0]
        else:
            img_batch = np.concatenate(all_img_pre, 0)

        cam = sess.run(self.tns_cam, feed_dict={self.tns_input: img_batch, self.tns_index: index_batch})

        return cam.squeeze()

    def compute_top_k_cams(self, image_batch):
        """ Compute the CAMs corresponding to the `top_k` predicted classes.

        :param image_batch: input images
        :return top_k_cams: the CAMs images for the first k classes
        :return top_k_preds: the indices of the first k predicted classes
        :return top_k_probs: the probabilities values for the first k predicted classes
        """
        all_img_pre = self.preprocess_input(image_batch)
        all_img_pre = np.concatenate(all_img_pre, 0)

        [top_k_preds, top_k_probs, top_k_cams] = sess.run([self.tns_top_k_pred,
                                                           self.tns_top_k_prob,
                                                           self.tns_top_k_cam], feed_dict={self.tns_input: all_img_pre})

        return top_k_cams, top_k_preds, top_k_probs


class Augmenter:

    def __init__(self, num_aug, augcamsz=(224, 224)):
        self.num_aug = num_aug
        self.augcamsz = augcamsz
        self.tns_angle = tf.placeholder("float", [num_aug])
        self.tns_shift = tf.placeholder("float", [num_aug, 2])
        self.tns_input_img = tf.placeholder("float", [1, augcamsz[0], augcamsz[1], 3])
        self.tns_img_batch = tf.placeholder("float", [num_aug, augcamsz[0], augcamsz[1], 3])

        # tensors for direct augmentation (input transformation)
        tns_img_exp = tf.tile(tf.expand_dims(self.tns_input_img[0], 0), [num_aug, 1, 1, 1])
        tns_rot_img = tf.contrib.image.rotate(tns_img_exp, self.tns_angle, interpolation="BILINEAR")
        self.tns_input_aug = tf.contrib.image.translate(tns_rot_img, self.tns_shift, interpolation="BILINEAR")

        # tensors for inverse augmentation (input anti-transformation)
        tns_shift_img_batch = tf.contrib.image.translate(self.tns_img_batch, self.tns_shift, interpolation="BILINEAR")
        self.tns_inverse_img_batch = tf.contrib.image.rotate(tns_shift_img_batch, self.tns_angle, interpolation="BILINEAR")

    def direct_augment(self, img, angles, shift):
        """ Apply rotation and shift to `img` according to the values in `angles` and `shift`.

        :param img: the input image
        :param angles: the magnitude of the rotation in radians
        :param shift: the magnitude of the shift
        :return img_aug: the transformed image
        """
        feed_dict = {self.tns_input_img: img[np.newaxis, :, :, :],
                     self.tns_angle: angles,
                     self.tns_shift: shift,
                     }
        img_aug = sess.run(self.tns_input_aug, feed_dict)

        return img_aug

    def inverse_augment(self, img_batch, angles, shift):
        """ Apply the inverse rotation and shift to `img_batch` according to the values in `angles` and `shift`.

        :param img_batch: a set of images to be anti-transformed
        :param angles: the magnitude of the rotatation in radians
        :param shift: the magnitude of the shift
        :return img_aug: the anti-transformed image
        """
        feed_dict = {self.tns_img_batch: img_batch,
                     self.tns_angle: -np.array(angles),
                     self.tns_shift: -np.array(shift),
                     }
        img_aug = sess.run(self.tns_input_aug, feed_dict)

        return img_aug


class Superresolution:

    def __init__(self, augmenter, learning_rate=0.001, camsz=(14, 14)):
        num_aug = augmenter.num_aug
        self.augmenter = augmenter
        augcamsz = self.augmenter.augcamsz

        # placeholder tensor for the batch of CAMs resulting from augmentation
        self.tns_cam_aug= tf.placeholder("float", [num_aug, camsz[0], camsz[1], 1])
        # placeholder tensors for the regularization coefficients
        self.tns_lmbda_eng = tf.placeholder("float", [1], name="lambda_eng")
        self.tns_lmbda_tv = tf.placeholder("float", [1], name="lambda_tv")
        # variable tensor for the target upsampled CAM
        self.tns_cam_full = tf.Variable(tf.zeros([1, augcamsz[0], augcamsz[1], 1]), name="cam_full")
        # augmentation parameters tensors
        tns_rot_cam = tf.contrib.image.rotate(tf.tile(self.tns_cam_full, [num_aug, 1, 1, 1]),
                                              augmenter.tns_angle,
                                              interpolation="BILINEAR")
        tns_aug = tf.contrib.image.translate(tns_rot_cam, augmenter.tns_shift, interpolation="BILINEAR")
        # tensor for the downsampling operator
        tns_Dv = tf.expand_dims(tf.image.resize(tns_aug, camsz, name="downsampling"), 0)
        # tensor for the gradient term
        tns_gradv = tf.image.image_gradients(self.tns_cam_full)

        # tensors for the functional terms
        tns_df = tf.reduce_sum(tf.squared_difference(tns_Dv, self.tns_cam_aug), name="data_fidelity")
        tns_tv = tf.reduce_sum(tf.add(tf.abs(tns_gradv[0]), tf.abs(tns_gradv[1])))
        tns_norm = tf.reduce_sum(tf.square(self.tns_cam_full))

        # loss definition
        self.tns_functional_tv = tf.add(tns_df, tf.scalar_mul(self.tns_lmbda_tv[0], tns_tv), name="loss_en_grad")
        self.tns_functional_mixed = tf.add(tf.scalar_mul(self.tns_lmbda_eng[0], tns_norm), self.tns_functional_tv)

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.minimizer_mixed = self.optimizer.minimize(self.tns_functional_mixed)
        self.tns_super_single = tf.image.resize(self.tns_cam_aug[0], augcamsz) # first CAM is the non-augmented one

        tns_img_batch = tf.image.resize(self.tns_cam_aug, augcamsz)
        tns_shift_img_batch = tf.contrib.image.translate(tns_img_batch,
                                                         self.augmenter.tns_shift,
                                                         interpolation="BILINEAR")
        self.tns_inverse_img_batch = tf.contrib.image.rotate(tns_shift_img_batch,
                                                             self.augmenter.tns_angle,
                                                             interpolation="BILINEAR")
        self.tns_max_aggr = tf.reduce_max(self.tns_inverse_img_batch, axis=0)
        self.tns_avg_aggr = tf.reduce_mean(self.tns_inverse_img_batch, axis=0)

        all_initializers = []
        all_initializers.append(self.tns_cam_full.initializer)
        for var in self.optimizer.variables():
            all_initializers.append(var.initializer)

        self.all_initializers = all_initializers


    def super_mixed(self, cams, angles, shift, lmbda_tv, lmbda_eng, niter=200):
        """ Compute the CAM solving the super resolution problem.

        :param cams: batch of CAMs
        :param angles: rotation values used to compute each CAM in `cams`
        :param shift: shift values used to compute each CAM in `cams`
        :param lmbda_tv: coefficient promoting the total variation
        :param lmbda_eng: coefficient promoting the energy of the gradient
        :param niter: number of iterations of the gradient descent
        :return cam_full_aug: the upsampled CAM resluting from super resolution aggregation
        """
        feed_dict = {self.tns_cam_aug: cams[:, :, :],
                     self.augmenter.tns_angle: angles,
                     self.augmenter.tns_shift: shift,
                     self.tns_lmbda_tv: [lmbda_tv],
                     self.tns_lmbda_eng: [lmbda_eng]
                     }

        sess.run(self.all_initializers)
        for ii in range(niter):
            _, func = sess.run([self.minimizer_mixed, self.tns_functional_mixed], feed_dict=feed_dict)
            print("{0:3d}/{1:3d} -- loss = {2:.5f}".format(ii+1, niter, func))
        cam_full_aug = sess.run(self.tns_cam_full)

        return cam_full_aug

    def super_single(self, cams):
        """ Compute the upsampled CAM with no aggregation.

        The first CAM in `cams` is the CAM with no augmentation

        :param cams: batch of CAMs
        :return cam_full: the upsampled CAM
        """
        cam_full = sess.run(self.tns_super_single, feed_dict={self.tns_cam_aug: cams})
        return cam_full

    def super_max(self, cams, angles, shift):
        """ Aggregation of `cams` using the max operator.

        CAMs are registered here, applying the anti-transformation using `angles` and `shift`

        :param cams: batch of CAMs
        :param angles: rotation values used to compute each CAM in `cams`
        :param shift: shift values used to compute each CAM in `cams`
        :return cam_full_aug: the upsampled CAM resulting from applying the max operator to the registered CAMs
        """
        feed_dict = {self.tns_cam_aug: cams[:, :, :],
                     self.augmenter.tns_angle: -angles,
                     self.augmenter.tns_shift: -shift
                     }

        cam_full_aug = sess.run(self.tns_max_aggr, feed_dict=feed_dict)
        return cam_full_aug

    def super_avg(self, cams, angles, shift):
        """ Aggregation of `cams` using the mean operator.

        CAMs are registered here, applying the anti-transformation using `angles` and `shift`

        :param cams: batch of CAMs
        :param angles: rotation values used to compute each CAM in `cams`
        :param shift: shift values used to compute each CAM in `cams`
        :return cam_full_aug: the upsampled CAM resulting from averaging the registered CAMs
        """
        feed_dict = {self.tns_cam_aug: cams[:, :, :],
                     self.augmenter.tns_angle: -angles,
                     self.augmenter.tns_shift: -shift
                     }

        cam_full_aug = sess.run(self.tns_avg_aggr, feed_dict=feed_dict)
        return cam_full_aug


    def overlay(self, cam, img, th=27, name=None):
        """ Overlay `cam` to `img`.

        The overlay of `cam` to `img` is done cutting away regions in `cam` below `th` for better visualization.

        :param cam: the class activation map
        :param img: the test image
        :param th: threshold for regions cut away from `cam`
        :param name: optional output filename
        :return o: the heatmap superimposed to `img`
        """
        # rotate color channels according to cv2
        background = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        foreground = np.uint8(cam / cam.max() * 255)
        foreground = cv2.applyColorMap(foreground, cv2.COLORMAP_JET)

        # mask the heatmap to remove near null regions
        gray = cv2.cvtColor(foreground, code=cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, thresh=th, maxval=255, type=cv2.THRESH_BINARY)
        mask = cv2.merge([mask, mask, mask])
        masked_foreground = cv2.bitwise_and(foreground, mask)

        # overlay heatmap to input image
        o = cv2.addWeighted(background, 0.6, masked_foreground, 0.4, 0)
        if name != None:
            cv2.imwrite("./imgs/" + name + ".png", o)

        return o
