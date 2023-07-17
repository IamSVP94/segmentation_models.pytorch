from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from svp_customs.utills import glob_search, get_random_colors

# SOURCE_DIR = Path('/home/vid/hdd/file/project/143-NLMK-DCA/Theme4Dim/labelmedataset/preprocess_synch/')
SOURCE_DIR = Path('/home/vid/hdd/file/project/143-NLMK-DCA/Theme4Dim/labelmedataset/TEST_DATASET/')
# RESULT_DIR = SOURCE_DIR.parent / f'{SOURCE_DIR.name}_ONNX_ready_tresholds'
RESULT_DIR = SOURCE_DIR.parent / f'{SOURCE_DIR.name}_ONNX_ready'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

model_common_dir = Path('/home/vid/hdd/file/project/143-NLMK-DCA/Theme4Dim/models/')
LABELS = ['smoke_cat_1', 'smoke_cat_2']
model_version, model_stem = 'R9.2', 'SMOKE_SEGMENTATION_R9.2_14072023_1280x736'
THRESHOLDS = [0.5, 0.5]  # THRESHOLDS here (output)
WINDOW_SIZE = (1280, 736)  # w,h format

colors = [(0, 0, 255), (0, 255, 0)]

MODEL_PATH = '/home/vid/hdd/projects/PycharmProjects/segmentation_models.pytorch_iamsvp94/svp_customs/lightning_logs/FPN_inceptionv4_FPN_inceptionv4_model/version_4/FPN_inceptionv4_R10_1280x736.onnx'

# config parameters from json
image_multiplier = 1 / 255.0
fp16 = True  # inference
image_mean_subtraction = [127.5, 127.5, 127.5]
blob_normalization_scale = 2.0
blob_mean_subtraction = [0.5, 0.5, 0.5]
blob_standard_deviation = [0.5, 0.5, 0.5]


def draw_mask(img, mask, colors=get_random_colors(3), thresholds=None):
    if thresholds:
        preds_thr = []
        for cl in range(mask.shape[-1]):
            bool_class_mask = np.where(mask[:, :, cl] > thresholds[cl], 1, 0)
            preds_thr.append(bool_class_mask)
        mask = np.stack(preds_thr).transpose(1, 2, 0)

    img_mask_pred = img.copy()
    for ch_idx in range(mask.shape[-1]):
        ch_pred = mask[:, :, ch_idx]
        confidences_pred = np.stack((ch_pred, ch_pred, ch_pred)).transpose(1, 2, 0)
        full_color = np.full(shape=confidences_pred.shape, fill_value=colors[ch_idx][::-1])
        img_mask_pred = img_mask_pred * (1.0 - confidences_pred) + full_color * confidences_pred
    return img_mask_pred


net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
if cv2.cuda.getCudaEnabledDeviceCount():
    device = 'cuda'
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    if fp16:
        device = 'cuda fp16'
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)  # fp16 here
else:
    device = 'cpu'
print(f"[INFO] Model is loaded on {device}!")

if __name__ == "__main__":
    imgs = glob_search(SOURCE_DIR)
    p_bar = tqdm(imgs, colour='yellow')
    for img_idx, img_path in enumerate(p_bar):
        p_bar.set_description(f'{img_path}')

        img = cv2.imread(str(img_path))
        orig_img_height, orig_img_width = img.shape[:2]
        scale_coef_h, scale_coef_w = orig_img_height / WINDOW_SIZE[1], orig_img_width / WINDOW_SIZE[0]

        # ''' ---------- blob cv2 preproccessing ----------
        input_blob_cv2 = cv2.dnn.blobFromImage(
            image=img,
            scalefactor=image_multiplier,  # "multiplier": 0.00392,
            size=WINDOW_SIZE,  # "spatial_size": [ 1920, 1088 ]
            mean=image_mean_subtraction,  # "mean_subtraction": [127.5, 127.5, 127.5]
            swapRB=True,  # "channels_order": "RGB"
            crop=None,  # "crop_after_resize": false
            ddepth=cv2.CV_32F  # "depth": "32F",
        )  # params from json
        # resize swap, mean, std - order operations
        input_blob_cv2 = input_blob_cv2 / np.asarray(blob_standard_deviation, dtype=np.float32).reshape(1, 3, 1, 1)
        # ========== /blob cv2 preproccessing ========== '''

        ''' ---------- blob torch preproccessing ----------
        test_image_src1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_image_src1 = cv2.resize(test_image_src1, WINDOW_SIZE, interpolation=cv2.INTER_NEAREST)

        input_range = [0, 1]
        if input_range is not None:
            if test_image_src1.max() > 1 and input_range[1] == 1:
                test_image_src1 = test_image_src1 / 255.0  # image_multiplier

        mean = [0.5, 0.5, 0.5]
        if mean is not None:
            mean = np.array(mean)
        test_image_src1 = test_image_src1 - mean  # image_mean_subtraction

        std = [0.5, 0.5, 0.5]
        if std is not None:
            std = np.array(std)
        test_image_src1 = test_image_src1 / std  # blob_standard_deviation

        input_blob_cv2 = np.transpose(test_image_src1, (2, 0, 1))
        input_blob_cv2 = input_blob_cv2[np.newaxis, ...].astype(np.float32)
        # ========== /blob torch preproccessing ========== '''
        # print(105, input_blob_cv2.shape)

        net.setInput(input_blob_cv2, scalefactor=blob_normalization_scale, mean=blob_mean_subtraction)
        # net.setInput(input_blob_cv2)
        pred = net.forward().squeeze(axis=0)  # Get model prediction
        pred = cv2.resize(pred.transpose(1, 2, 0), (orig_img_width, orig_img_height), interpolation=cv2.INTER_LINEAR)

        with_mask = draw_mask(img, pred, colors=colors, thresholds=THRESHOLDS)
        # with_mask = InferenceSmokeDataset.draw_mask(img, pred, colors=colors)
        with_mask_path = RESULT_DIR / img_path.name
        cv2.imwrite(str(with_mask_path), with_mask)
