from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


from matplotlib import pyplot as plt, pyplot
from sklearn.decomposition import PCA

preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions
sys.setrecursionlimit(40000)

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", dest="num_rois",
                  help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--meta", action='store_true', default=False, help="meta missed.")
parser.add_option("--config_filename", dest="config_filename", help=
"Location to read the metadata related to the training (generated when training).",
                  default="config.pickle")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
                  default="pascal_voc"),

(options, args) = parser.parse_args()
features_name_file = "festures_model_rpn_original.npy"

if not options.test_path:  # if filename is not given
    parser.error('Error: path to test data must be specified. Pass --path to command line')

if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
    from keras_frcnn.simple_parser import get_data
else:
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

config_output_filename = options.config_filename
'''
with open(config_output_filename, 'r') as f_in:
	C = pickle.load(f_in)
'''
C = config.Config()
# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

if C.network == 'resnet50':
    import frcnn.resnet as nn
elif C.network == 'vgg':
    import frcnn.vgg as nn
elif C.network == 'resnet101':
    import frcnn.resnet101 as nn

img_path = options.test_path

print('Loading ontology')
f = open(frcnn/pascalPartOntology.csv', 'r')
ontology = {}
for l in f.readlines():
    classes = l.split(',')
    classes[-1] = classes[-1][:-1]
    ontology[classes[0]] = classes[1:]
objects = list(ontology.keys())

print('Parsing annotation files')


def deprocess_image(x):
   
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def format_img(img, C):
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        f = img_min_side / width
        new_height = int(f * height)
        new_width = int(img_min_side)
    else:
        f = img_min_side / height
        new_width = int(f * width)
        new_height = int(img_min_side)
    fx = width / float(new_width)
    fy = height / float(new_height)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, fx, fy


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_2(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


def load_image_prova(path, width, height, preprocess=True):
    """Load and preprocess image."""
    x = keras.preprocessing.image.load_img(path, target_size=(width, height))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    return x


# class_mapping = C.class_mapping
all_imgs, _, class_mapping = get_data(options.test_path, options.meta)

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {k.lower(): v for k, v in class_mapping.items()}

print(class_mapping)

C.num_rois = int(options.num_rois)

if K.common.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (1024, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, 1024)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifierEvaluateTsnet(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping),
                                        trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)
base_net_weights = join("Experiment(FAS-922)", "155_model_all_PASCAL_VOC__1.9980390360329519.hdf5")
model_rpn.load_weights(base_net_weights, by_name=True)
model_classifier.load_weights(base_net_weights, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')
model_classifier.summary()

test_imgs = [s for s in all_imgs if s['imageset'] == 'val']

if not os.path.exists(features_name_file):
    random_element = random.sample(test_imgs, 300)

    img_path = 'test/JPEGImages/2007_000175.jpg'
    # if s['imageset'] == 'test'
    img_size = (299, 299)
    print("test_len" + len(test_imgs).__str__())
    # test_imgs = [s for s in all_imgs if  "2007_000027" in s['filepath']]
    # test_imgs=[ s for s in test_imgs if "2007_000175" in s['filepath']]
    all_dets = []
    for idx, img_data in enumerate(test_imgs):
        img_path = img_data['filepath']
        # print(img_path)
        if idx % 500 == 0:
            print(idx)
        # img_path = 'test/JPEGImages/2007_000175.jpg'
        # img_array = get_img_array(img_path, size=img_size)
        img = cv2.imread(img_path)

        # img = cv2.imread(img_path)
        X, ratio = format_img_2(img, C)
        if K.common.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)
        # print("Predicted:", decode_predictions(preds, top=1)[0])

        R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
        features = {}
        RoIs_array = {}
        features_array = {}
        indices_array = {}
        RoIs_array = []
        indices_array = []
        features_array = []
        label_array = []
        bboxes_array = []
        predictions_array = []

        # model_classifier.summary()

        # model_rpn.summary()

        axis_x = 8
        axis_y = 4

        if X2 is not None:
            R = X2[0]
            index = 0

            for jk in range(R.shape[0] // C.num_rois + 1):
                ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
                if ROIs.shape[1] == 0:
                    break

                if jk == R.shape[0] // C.num_rois:
                    # pad R
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:, :curr_shape[1], :] = ROIs
                    ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                    ROIs = ROIs_padded

                [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

                for ii in range(P_cls.shape[1]):
                    index += 1

                    if index > (Y1.shape[1]):
                        continue

                    # cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
                    # print(ii)
                    try:
                        cls_name = list((class_mapping.keys()))[
                            list((class_mapping.values())).index(np.argmax(Y1[0][index - 1]))]

                    except:
                        print("dfdf")
                    if cls_name not in bboxes:
                        bboxes[cls_name] = []
                        probs[cls_name] = []
                        features[cls_name] = []

                    (x, y, w, h) = ROIs[0, ii, :]

                    cls_num = np.argmax(Y1[0][index - 1])
                    # cls_num = 0
                    try:
                        (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                        tx /= C.classifier_regr_std[0]
                        ty /= C.classifier_regr_std[1]
                        tw /= C.classifier_regr_std[2]
                        th /= C.classifier_regr_std[3]
                        x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                    except:
                        pass

                    bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
                    # probs[cls_name].append(np.max(P_cls[0, ii, :]))
                    features[cls_name].append(P_cls[0, ii, :])

                    features_array.append(F)
                    RoIs_array.append(ROIs)
                    indices_array.append(ii)
                    bboxes_array.append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
                    predictions_array.append([cls_name, P_cls[0, ii, :]])

                img = cv2.imread(img_path)
                for key in bboxes:
                    bbox = np.array(bboxes[key])
                    for j in range(bbox.shape[0]):
                        (x1, y1, x2, y2) = bbox[j, :]
                        det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key,
                               'features': features[key][j], 'image_path': img_path}
                        all_dets.append(det)

                bboxes = {}
                features = {}

    save(features_name_file, all_dets)

print("..tsne training...")
all_dets = load(features_name_file, allow_pickle=True)
all_dets_random = all_dets
all_dets_random = [f for f in all_dets_random if f['class'].lower() in objects]
pca = PCA(n_components=50)
X_pca = pca.fit_transform(np.array([f['features'] for f in all_dets_random]).reshape(-1, 1 * 1 * 2048))

for p in [50]:
    tsne = TSNE(n_jobs=-1, perplexity=p, n_iter=1000).fit_transform(X_pca)
    tx, ty = tsne[:, 0], tsne[:, 1]
    df_subset = {}
    df_subset['tsne-2d-one'] = tsne[:, 0]
    df_subset['tsne-2d-two'] = tsne[:, 1]
    df_subset['class'] = [f['class'].capitalize() for f in all_dets_random]

    # tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    # ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
    # initialize a matplotlib plot

    fig = plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="class",
        palette=sns.color_palette("hls", len(objects)),
        data=df_subset,
        legend="full",
        alpha=0.3
    )

    # plt.figure(dpi=400)

    fig.savefig("feature_tsnet_" + p.__str__() + ".png")
    fig.savefig("feature_tsnet_" + p.__str__() + ".pdf", bbox_inches='tight')
    print("saved")
    plt.show()


# Compute the coordinates of the image on the plot
def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center,
    # compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y
