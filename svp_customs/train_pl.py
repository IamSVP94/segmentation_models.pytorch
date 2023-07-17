import cv2
import torch
from clearml import Task
from pathlib import Path
import albumentations as albu
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from svp_customs.pt_train_utills import CustomSmokeDataset, SmokeModel, MetricSMPCallback

pl.seed_everything(2)

EXPERIMENT_NAME = 'FPN_inceptionv4'
MODEL_VERSION = 'R10'

task = Task.init(
    project_name='143-NLMK-DCA_Theme4Dim',
    task_name=EXPERIMENT_NAME,
    tags=['segmentation', 'Smoke_segmentation', 'pl', MODEL_VERSION, 'self_augmentation_v2'],
)

# TODO: resize before train
train_dir = '/home/vid/hdd/file/project/143-NLMK-DCA/Theme4Dim/labelmedataset/TRAIN_DATASET/'
val_dir = '/home/vid/hdd/file/project/143-NLMK-DCA/Theme4Dim/labelmedataset/VAL_DATASET/'
test_dir = '/home/vid/hdd/file/project/143-NLMK-DCA/Theme4Dim/labelmedataset/TEST_DATASET/'

# TODO: add yaml configurator
arch = 'FPN'
ENCODER = 'inceptionv4'
# arch = 'UnetPlusPlus'
# ENCODER = 'resnet34'

ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['smoke_cat_1', 'smoke_cat_2']  # 2 classes + 1 background
colors = [(0, 0, 255), (0, 255, 0)]

ACTIVATION = None

# input_width, input_height = 1920, 1088
input_width, input_height = 1280, 736
# input_width, input_height = 512, 288

'''
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
'''

# '''
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


# '''


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# TODO: write fix preprocessing for easy subsequent using with ONNX
preprocessing_fn = smp.encoders.get_preprocessing_fn('inceptionv4', 'imagenet')  # fix std, mean = [0.5,0.5,0.5]
# preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = CustomSmokeDataset(
    dataset_dir=train_dir, input_width=input_width, input_height=input_height,
    classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn),
    augmentation=train_transform,
)

val_dataset = CustomSmokeDataset(
    dataset_dir=val_dir, input_width=input_width, input_height=input_height,
    classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn),
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=15)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=15)

mode = smp.losses.BINARY_MODE

# criterion = [
#     smp.losses.DiceLoss(from_logits=True, log_loss=True, mode=mode, ),
#     smp.losses.SoftBCEWithLogitsLoss(reduction='mean'),
#     torch.nn.BCEWithLogitsLoss(reduction='mean'),
# ]
# TODO: составной лосс
criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

metrics_callback = MetricSMPCallback(
    metrics={
        'iou': smp.metrics.iou_score,
        'f1_score': smp.metrics.f1_score,
    },
    threshold=[0.5, 0.5], reduction='macro', classes_separately=True,
    activation='sigmoid', mode=mode, colors=colors,
    n_img_check_per_epoch_validation=10,
    n_img_check_per_epoch_train=2,
    n_img_check_per_epoch_save=True,
    log_img=False, save_img=True,
)

start_learning_rate = 1e-3

# n_steps, times = 60, 3  # every "n_steps" steps "times" times
# scheduler_steps = {n_steps * (i + 1) for i in range(times)}
# scheduler_steps.add(epochs - 10)
# print('scheduler_steps:', scheduler_steps)
# scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.5, milestones=scheduler_steps, verbose=True)
# scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler_steplr)

# TODO: add scheduler in SmokeModel
model = SmokeModel(
    arch=arch,
    encoder_name=ENCODER,
    activation=ACTIVATION,
    loss_fn=criterion,
    in_channels=3,  # RGB
    classes=CLASSES,
    start_learning_rate=start_learning_rate,
)

tb_logger = TensorBoardLogger('lightning_logs', name=f'{arch}_{ENCODER}_{EXPERIMENT_NAME}_model')
lr_monitor = LearningRateMonitor(logging_interval='epoch')
best_iou_saver = ModelCheckpoint(
    monitor='iou/validation_total',
    mode='max', save_top_k=1, save_last=True,
    filename='epoch={epoch:02d}-iou_validation_total={iou/validation_total:.4f}',
    auto_insert_metric_name=False,
)

trainer = pl.Trainer(
    max_epochs=300,
    accelerator='cuda',
    devices=-1,
    num_sanity_val_steps=0,
    logger=tb_logger,
    callbacks=[lr_monitor, metrics_callback, best_iou_saver],
)

weights = '/home/vid/hdd/projects/PycharmProjects/segmentation_models.pytorch_iamsvp94/svp_customs/lightning_logs/FPN_inceptionv4_FPN_inceptionv4_model/version_3/checkpoints/epoch=epoch=61-iou_validation=iou/validation_total=0.6657.ckpt'

trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    ckpt_path=weights,
)

test_dataset = CustomSmokeDataset(
    dataset_dir=test_dir, input_width=input_width, input_height=input_height,
    classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn),
)
test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=15)
test_metrics = trainer.test(model, dataloaders=test_loader, ckpt_path=best_iou_saver.best_model_path, verbose=True)

logger_weights_dir = Path(best_iou_saver.best_model_path).parent.parent
ONNX_PATH = logger_weights_dir / f'{arch}_{ENCODER}_{MODEL_VERSION}_{input_width}x{input_height}.onnx'

model.save_onnx_best(
    weights=best_iou_saver.best_model_path,
    window_size=(input_width, input_height),
    onnx_path=ONNX_PATH,
)
