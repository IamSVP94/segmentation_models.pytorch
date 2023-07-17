from pathlib import Path

import albumentations as albu
import cv2
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import segmentation_models_pytorch as smp
from svp_customs.pt_train_utills import InferenceSmokeDataset, SmokeModel, MetricSMPCallback
from svp_customs.utills import plt_show_img

pl.seed_everything(2)

EXPERIMENT_NAME = 'FPN_inceptionv4'

arch = 'FPN'
ENCODER = 'inceptionv4'

ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['smoke_cat_1', 'smoke_cat_2']  # 2 classes + 1 background
colors = [(0, 0, 255), (0, 255, 0)]

ACTIVATION = 'sigmoid'

input_width, input_height = 1280, 736


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


preprocessing_fn = smp.encoders.get_preprocessing_fn('inceptionv4', 'imagenet')  # fix std, mean = [0.5,0.5,0.5]

inference_dir = Path('/home/vid/hdd/file/project/143-NLMK-DCA/Theme4Dim/labelmedataset/TEST_DATASET/')
new_save_dir = inference_dir.parent / f'{inference_dir.stem}_ready_thresholds/'

weights = '/home/vid/hdd/projects/PycharmProjects/segmentation_models.pytorch_iamsvp94/svp_customs/lightning_logs/FPN_inceptionv4_FPN_inceptionv4_model/version_4/checkpoints/epoch=125-iou_validation_total=0.6804.ckpt'
model = SmokeModel(
    arch=arch,
    encoder_name=ENCODER,
    activation=ACTIVATION,
    classes=CLASSES,
    # weights=weights,
)

inference_dataset = InferenceSmokeDataset(
    dataset_dir=inference_dir, input_width=input_width, input_height=input_height,
    classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn),
)
inference_loader = DataLoader(inference_dataset, batch_size=15, shuffle=False, num_workers=10, drop_last=False)
trainer = pl.Trainer(accelerator='cuda', devices=-1, num_sanity_val_steps=0)

# TODO: change it because OOM
preds = trainer.predict(model, ckpt_path=weights, dataloaders=inference_loader, return_predictions=True)
preds = torch.permute(torch.cat(preds, 0), (0, 2, 3, 1)).cpu().detach().numpy()

new_save_dir.mkdir(parents=True, exist_ok=True)
for img_path, pred in tqdm(zip(inference_dataset.imgs, preds), total=len(inference_dataset)):
    image = cv2.imread(str(img_path))
    if image.shape[:2] != (input_height, input_width):
        image = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_NEAREST)
    with_mask = InferenceSmokeDataset.draw_mask(image, pred, colors=colors, thresholds=[0.5, 0.5])
    with_mask_path = new_save_dir / img_path.name
    cv2.imwrite(str(with_mask_path), with_mask)
