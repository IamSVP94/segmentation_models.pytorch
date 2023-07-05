import os
from pathlib import Path



os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import cv2
import matplotlib.pyplot as plt
import albumentations as albu

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import torch
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
import pytorch_lightning as pl

from clearml import Task, Logger

task = Task.init(
    project_name='ClearML_smoke_segmentation',
    task_name='smoke_segmentation_v2',
    tags=['segmentation', 'Smoke_segmentation']
)
log = Logger.current_logger()

SHOW_METRIC_PLOT = False

x_train_dir = DATASET_ROOT / 'train'
y_train_dir = DATASET_ROOT / 'train_annot'

x_valid_dir = DATASET_ROOT / 'val'
y_valid_dir = DATASET_ROOT / 'val_annot'

x_test_dir = '/home/vid/hdd/file/project/143-NLMK-DCA/Theme4Dim/labelmedataset/TEST_DATASET/img/'
y_test_dir = '/home/vid/hdd/file/project/143-NLMK-DCA/Theme4Dim/labelmedataset/TEST_DATASET/annot/'

input_width, input_height = 1920, 1088


# task.upload_artifact(name='train_list', artifact_object=glob_search(x_train_dir))
# task.upload_artifact(name='val_list', artifact_object=glob_search(x_valid_dir))
# task.upload_artifact(name='test_list', artifact_object=glob_search(x_test_dir))

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1,
                              border_mode=cv2.BORDER_REFLECT_101),

        # albu.CropNonEmptyMaskIfExists(height=1088, width=1088, always_apply=True, p=1.0),
        albu.CropNonEmptyMaskIfExists(height=input_height, width=input_width, always_apply=True, p=1.0),

        albu.OneOf(
            [
                albu.RandomBrightness(p=1.0, limit=0.1),
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

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1, hue_shift_limit=0, sat_shift_limit=30, val_shift_limit=20, ),
            ],
            p=0.5,
        ),
    ]
    return albu.Compose(train_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


class Dataset(BaseDataset):
    CLASSES = ['unlabelled', 'smoke_cat_1', 'smoke_cat_2']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        # self.images_fps[i] = '/home/vid/hdd/file/project/NLMK_SILA/Тема#4Дым/imgs/20230410_не_сработала_НС_imgs/09_43_37_753_45157_o.jpg'
        # print(202, self.images_fps[i])
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        image = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (input_width, input_height), interpolation=cv2.INTER_NEAREST)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        # print()
        # print(222, image)
        return image, mask

    def __len__(self):
        return len(self.ids)


''' augmentation checking
img_path = '/home/vid/hdd/file/project/143-NLMK-DCA/Theme4Dim/labelmedataset/TEST_DATASET/img/14_52_53_661_50272_o.png'
annot_path = '/home/vid/hdd/file/project/143-NLMK-DCA/Theme4Dim/labelmedataset/TEST_DATASET/annot/14_52_53_661_50272_o.png'
img = cv2.imread(str(img_path))
annot = cv2.imread(str(annot_path))
plt_show_img(img, title='orig', coef=None)
for i in range(10):
    aug = get_training_augmentation()
    img_aug = aug(image=img, mask=annot)['image']
    plt_show_img(img_aug, title=i, coef=None)
exit()
# '''

# TRAIN
ENCODER = "inceptionv4"  # исходный
# ENCODER = "resnet101"  #
# ENCODER = "resnext101_32x4d"  # это трансформер

ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['smoke_cat_1', 'smoke_cat_2']
ACTIVATION = None  # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'
MAX_EPOCHS = 300

model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=None,
    classes=len(CLASSES),
    activation=ACTIVATION,
)
# model = torch.load(
#     '/home/vid/hdd/projects/PycharmProjects/segmentation_models.pytorch-0.1.3/temp/FPN_inceptionv4_best_maxiou_0.46_(6).pth')

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)
test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=15)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=15)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=15)

weighted = torch.tensor([1, 3])
loss = smp.losses.SoftBCEWithLogitsLoss(weight=torch.reshape(weighted, (1, len(CLASSES), 1, 1)))

metrics = [
    smp.metrics.iou_score(class_weights=weighted, reduction='weighted'),
    smp.metrics.f1_score(class_weights=weighted, reduction='weighted'),
]

optimizer = torch.optim.AdamW([dict(
    params=model.parameters(),
    lr=1e-5 * 5,
    weight_decay=1e-4,
)])

train_epoch = smp.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

max_fscore = 0.0
max_iou_score = 0.0

valid_fscore_metrics, valid_iou_metrics = [], []
train_fscore_metrics, train_iou_metrics = [], []
idxs = []
for i in range(0, MAX_EPOCHS):
    print(f'\nEpoch: {i}')
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # ''' ClearML logging
    for (v_k, v_v), (t_k, t_v) in zip(valid_logs.items(), train_logs.items()):
        log.report_scalar(v_k, "Val", iteration=i, value=v_v)
        log.report_scalar(t_k, "Train", iteration=i, value=t_v)
    else:
        log.report_scalar('learning_rate', 'lr', iteration=i, value=optimizer.param_groups[0]['lr'])
    # /ClearML logging '''

    if max_fscore < valid_logs['fscore_batch']:
        max_fscore = valid_logs['fscore_batch']
        torch.save(model, f'./{model.__class__.__name__}_{ENCODER}_best_maxfscore_{round(max_fscore, 2)}_({i}).pth')
        print(f'Model saved! (maxfscore = {max_fscore})')

    if max_iou_score < valid_logs['iou_score']:
        max_iou_score = valid_logs['iou_score']
        torch.save(model, f'./{model.__class__.__name__}_{ENCODER}_best_maxiou_{round(max_iou_score, 2)}_({i}).pth')
        print(f'Model saved! (maxiou = {max_iou_score})')

    if i in [75, 150]:
        optimizer.param_groups[0]['lr'] *= 0.1
        print(f'Decrease decoder learning rate to {optimizer.param_groups[0]["lr"]}!')

    if SHOW_METRIC_PLOT:
        valid_iou_metrics.append(valid_logs['iou_score'])  # for iou val plot
        valid_fscore_metrics.append(valid_logs['fscore_batch'])  # for fs val plot
        train_iou_metrics.append(train_logs['iou_score'])  # for iou train plot
        train_fscore_metrics.append(train_logs['fscore_batch'])  # for fs train plot
        idxs.append(i)
        fig, (ax_iou, ax_fs) = plt.subplots(nrows=1, ncols=2)
        if i != 0:
            ax_iou.plot(idxs, train_iou_metrics, label='train')
            ax_iou.plot(idxs, valid_iou_metrics, label='val')
            ax_iou.set_title('IoU')

            ax_fs.plot(idxs, train_fscore_metrics, label='train')
            ax_fs.plot(idxs, valid_fscore_metrics, label='val')
            ax_fs.set_title('F1 score')

            plt.legend()
            plt.show()
else:
    test_logs = valid_epoch.run(test_loader)
    for (t_k, t_v) in test_logs.items():
        log.report_scalar(t_k, "Test", iteration=i, value=t_v)

    torch.save(model, f'./{model.__class__.__name__}_{ENCODER}_last_{i}_model.pth')  # save last checkpoint
    task.close()
