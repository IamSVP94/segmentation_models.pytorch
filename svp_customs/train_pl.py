import cv2
import albumentations as albu
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from svp_customs.constants import PROJECT_DIR
from svp_customs.pt_train_utills import CustomSmokeDataset, SmokeModel, MetricSMPCallback
from svp_customs.utills import plt_show_img
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from clearml import Task

pl.seed_everything(2)

EXPERIMENT_NAME = 'ClearML testing'
MODEL_VERSION = 'R10'

# task = Task.init(
#     project_name='143-NLMK-DCA_Theme4Dim',
#     task_name=EXPERIMENT_NAME,
#     tags=['segmentation', 'Smoke_segmentation', 'pl', MODEL_VERSION],
# )

# TODO: resize на стадии до разметки или при составлении папки датасета!
train_dir = '/home/vid/hdd/file/project/143-NLMK-DCA/Theme4Dim/labelmedataset/TRAIN_DATASET/'
val_dir = '/home/vid/hdd/file/project/143-NLMK-DCA/Theme4Dim/labelmedataset/TEST_DATASET/img+jsons/'
# train_dir = '/home/vid/hdd/file/project/143-NLMK-DCA/Theme4Dim/labelmedataset/faketrain/'
# val_dir = '/home/vid/hdd/file/project/143-NLMK-DCA/Theme4Dim/labelmedataset/faketest/'

# TRAIN
arch = 'FPN'
ENCODER = 'inceptionv4'  # исходный

ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['smoke_cat_1', 'smoke_cat_2']  # 2 classes + 1 background
colors = [(0, 0, 255), (0, 255, 0)]

ACTIVATION = None  # could be None for logits or 'softmax2d' for multicalss segmentation

# input_width, input_height = 1920, 1088
# input_width, input_height = 1280, 736
input_width, input_height = 512, 288

train_transform = [
    albu.HorizontalFlip(p=0.5),
    albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
    # albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),

    albu.CropNonEmptyMaskIfExists(height=input_height, width=input_width, always_apply=True, p=1.0),

    # albu.Resize(height=256, width=256, always_apply=True),
    # albu.IAAAdditiveGaussianNoise(p=0.2),
    # albu.IAAPerspective(p=0.5),
    albu.OneOf(
        [
            albu.CLAHE(p=1),
            albu.RandomBrightness(p=1),
            albu.RandomGamma(p=1),
        ],
        p=0.9,
    ),
    albu.OneOf(
        [
            # albu.IAASharpen(p=1),
            albu.Blur(blur_limit=3, p=1),
            albu.MotionBlur(blur_limit=3, p=1),
        ],
        p=0.9,
    ),
    albu.OneOf(
        [
            albu.RandomContrast(p=1),
            albu.HueSaturationValue(p=1),
        ],
        p=0.9,
    ),
]

train_transform = []
'''
train_transform = [
    albu.HorizontalFlip(p=0.5),
    albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=cv2.BORDER_REFLECT_101),
    albu.CropNonEmptyMaskIfExists(height=input_height, width=input_width, always_apply=True, p=1.0),
    albu.OneOf(
        [
            albu.RandomBrightnessContrast(p=1.0, brightness_limit=0.1),
            albu.RandomGamma(p=1),
        ],
        p=0.5,
    ),
    albu.OneOf(
        [
            albu.GridDistortion(p=1),
            albu.ElasticTransform(p=1),
        ],
        p=0.5,
    ),
    albu.OneOf(
        [
            albu.Sharpen(p=1),
            albu.Blur(blur_limit=3, p=1),
            albu.MotionBlur(blur_limit=3, p=1),
        ],
        p=0.9,
    ),
    albu.HueSaturationValue(p=0.5, hue_shift_limit=0, sat_shift_limit=30, val_shift_limit=20, ),
]
'''


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = CustomSmokeDataset(
    dataset_dir=train_dir,
    input_width=input_width, input_height=input_height,
    classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn),
    augmentation=train_transform,
)

val_dataset = CustomSmokeDataset(
    dataset_dir=train_dir,
    input_width=input_width, input_height=input_height,
    classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn),
)

# window size = 512, 288
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=15)
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=15)

# mode = smp.losses.MULTILABEL_MODE
# mode = smp.losses.MULTICLASS_MODE
mode = smp.losses.BINARY_MODE

# from old version
# weighted = torch.tensor([1, 10])
# loss = smp.utils.losses.BCEWithLogitsLoss(weight=torch.reshape(weighted, (1, len(CLASSES), 1, 1)))

# criterion = [
#     smp.losses.DiceLoss(from_logits=True, log_loss=True, mode=mode, ),
#     smp.losses.SoftBCEWithLogitsLoss(reduction='mean'),
#     torch.nn.BCEWithLogitsLoss(reduction='mean'),
# ]
criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

metrics_callback = MetricSMPCallback(
    metrics={
        'iou': smp.metrics.iou_score,
        'f1_score': smp.metrics.f1_score,
    },
    threshold=0.5,
    activation='sigmoid', mode=mode, colors=colors,
    n_img_check_per_epoch_validation=10,
    n_img_check_per_epoch_train=2,
    n_img_check_per_epoch_save=True,
    log=True, save_img=True,
)

start_learning_rate = 1e-3

model = SmokeModel(
    arch=arch,
    encoder_name=ENCODER,
    activation=ACTIVATION,
    loss_fn=criterion,
    in_channels=3,  # RGB
    out_classes=len(CLASSES),
    start_learning_rate=start_learning_rate,
)

tb_logger = TensorBoardLogger('lightning_logs', name=f'{arch}_{ENCODER}_{EXPERIMENT_NAME}_model')

lr_monitor = LearningRateMonitor(logging_interval='epoch')

best_iou_saver = ModelCheckpoint(
    monitor='iou/validation',
    mode='max',
    save_top_k=1,
    save_last=True,
    filename='{epoch:02d}-{step:02d}-{"iou/validation":.4f}',
)

trainer = pl.Trainer(
    max_epochs=10,
    accelerator='cuda', devices=-1, num_sanity_val_steps=0, logger=tb_logger,
    callbacks=[lr_monitor, metrics_callback, best_iou_saver],
)

trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

# valid_metrics = trainer.validate(model, dataloaders=val_loader, verbose=False)
# print(113, valid_metrics)

# save bst model like ONNX
# model_path_pth = f'/home/vid/hdd/projects/PycharmProjects/open-metric-learning/temp/cashbox/oml7_{model.arch}_{epochs}_CustomLoss_{input_size[0]}x{input_size[1]}_{model.feat_dim}_distrib.pth'
# trainer.save_checkpoint(filepath=model_path_pth, weights_only=True)  # we don't pass loaders to .fit() in DDP
#
# model_path_onnx = Path(model_path_pth).with_suffix(f'.onnx')
# x = torch.randn(1, 3, input_size[1], input_size[0], requires_grad=True)
# torch.onnx.export(pl_model, x, str(model_path_onnx),
#                   export_params=True, verbose=True,
#                   opset_version=11,
#                   do_constant_folding=True,
#                   input_names=['input'],  # the model's input names
#                   output_names=['output'],
#                   )
