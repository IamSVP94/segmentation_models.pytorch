import json
import random
from pathlib import Path
from typing import Tuple, Union, List

import cv2
import numpy as np
from matplotlib import pyplot as plt
from svp_customs.constants import OPENCV_CONFIG


def get_random_colors(n=1):
    colors = []
    for i in range(n):
        randomcolor = (random.randint(0, 150), random.randint(50, 200), random.randint(50, 200))
        colors.append(randomcolor)
    return colors


def max_show_img_size_reshape(img, max_show_img_size, return_coef=False):  # h,w format
    img_c = img.copy()
    h, w = img_c.shape[:2]
    coef = 1
    if h > max_show_img_size[0] or w > max_show_img_size[1]:
        h_coef = h / max_show_img_size[0]
        w_coef = w / max_show_img_size[1]
        if h_coef < w_coef:  # save the biggest side
            new_img_width = max_show_img_size[1]
            coef = w / new_img_width
            new_img_height = h / coef
        else:
            new_img_height = max_show_img_size[0]
            coef = h / new_img_height
            new_img_width = w / coef
        new_img_height, new_img_width = map(int, [new_img_height, new_img_width])
        img_c = cv2.resize(img_c.astype(np.uint8), (new_img_width, new_img_height), interpolation=cv2.INTER_LINEAR)
    if return_coef:
        return img_c, coef
    return img_c


def plt_show_img(img,
                 title: str = None,
                 coef=None,
                 mode: str = 'plt',
                 max_img_size: Tuple[str] = (900, 900),
                 save_path: Union[str, Path] = None) -> None:
    if isinstance(img, (str, Path)):
        img = cv2.imread(str(img))
    img = img.astype(np.uint8)
    img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if mode == 'plt' else img
    if coef:
        img_show = img_show * coef
    img_show = img_show.copy().astype(np.uint8)
    title = str(title) if title is not None else 'image'
    if mode == 'plt':
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(img_show)
        if title:
            ax.set_title(title)
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
        fig.show()
    elif mode == 'cv2':
        if max_img_size is not None:
            img_show = max_show_img_size_reshape(img_show, max_img_size).astype(np.uint8)
        cv2.imshow(title, img_show)
        cv2.waitKey(0)


def prepare_img_blob(image,
                     scalefactor=OPENCV_CONFIG['image_multiplier'],  # "multiplier": 0.00392,
                     size=OPENCV_CONFIG['image_spatial_size'],  # "spatial_size": [ 1920, 1088 ]
                     mean=OPENCV_CONFIG['image_mean_subtraction'],  # "mean_subtraction": [127.5, 127.5, 127.5]
                     swapRB=OPENCV_CONFIG['swapRB'],  # "channels_order": "RGB"
                     crop=OPENCV_CONFIG['crop'],  # "crop_after_resize": false
                     ddepth=OPENCV_CONFIG['ddepth'],  # "depth": "32F",
                     ):
    input_blob_cv2 = cv2.dnn.blobFromImage(
        image=image,
        scalefactor=scalefactor,  # "multiplier": 0.00392,
        size=size,  # "spatial_size": [ 1920, 1088 ]
        mean=mean,  # "mean_subtraction": [ 0.5, 0.5, 0.5 ] ???
        swapRB=swapRB,  # "channels_order": "RGB"
        crop=crop,  # "crop_after_resize": false
        ddepth=ddepth,  # "depth": "32F",
    )  # params from json
    input_blob_cv2_cv2 = input_blob_cv2 / np.asarray(OPENCV_CONFIG['blob_standard_deviation'],
                                                     dtype=np.float32).reshape(1, 3, 1, 1)
    return input_blob_cv2


def make_mask_from_json_old(json_path, classes=("smoke_cat_1", "smoke_cat_2")):
    with open(json_path, 'r') as label_json:
        json_txt = json.load(label_json)

    orig_h, orig_w = json_txt["imageHeight"], json_txt["imageWidth"]
    labels = json_txt["shapes"]
    mask = np.zeros((orig_h, orig_w, len(classes) + 1), dtype=np.uint8)

    for l in labels:
        class_index = classes.index(l["label"]) + 1
        points = l["points"]

        if l["shape_type"] == "polygon" or l["shape_type"] == "linestrip":
            color = [0] * len(classes)
            color[class_index] = 1

            contour = [np.array(points, dtype=np.int32)]
            cv2.drawContours(mask, [contour[0]], 0, color, -1)
        elif l["shape_type"] == "rectangle":
            cv2.rectangle(mask,
                          (int(points[0][0]), int(points[0][1])),
                          (int(points[1][0]), int(points[1][1])),
                          (class_index, class_index, class_index), -1)
    mask = mask[:, :, 1:]  # because we do not need "unlabelled" class
    return np.moveaxis(mask, -1, 0)


def glob_search(directories: Union[str, Path, List[str], List[Path]],
                pattern: str = '**/*',
                formats: Union[List[str], Tuple[str], str] = ('png', 'jpg', 'jpeg'),
                shuffle: bool = False,
                seed: int = 2,
                sort: bool = False,
                exception_if_empty=False):
    if isinstance(directories, (str, Path)):
        directories = [Path(directories)]
    files = []
    for directory in directories:
        if isinstance(directory, (str)):
            directory = Path(directory)
        if formats:
            if formats == '*':
                files.extend(directory.glob(f'{pattern}.{formats}'))
            else:
                for format in formats:
                    files.extend(directory.glob(f'{pattern}.{format.lower()}'))
                    files.extend(directory.glob(f'{pattern}.{format.upper()}'))
                    files.extend(directory.glob(f'{pattern}.{format.capitalize()}'))
        else:
            files.extend(directory.glob(f'{pattern}'))
    if exception_if_empty:
        if not len(files):
            raise Exception(f'There are no such files!')
    if shuffle:
        random.Random(seed).shuffle(files)
    if sort:
        files = sorted(files)
    return files
