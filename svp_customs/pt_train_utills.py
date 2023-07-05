import json
import cv2
from typing import Tuple
import random
import albumentations as albu
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset as BaseDataset
from svp_customs.utills import glob_search, prepare_img_blob, plt_show_img, get_random_color
from albumentations.pytorch import ToTensorV2 as ToTensor


class CustomSmokeDataset(BaseDataset):
    def __init__(
            self,
            dataset_dir,
            input_width=1920,
            input_height=1088,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.imgs = glob_search(dataset_dir, sort=True, exception_if_empty=True)
        self.masks = []
        for img_idx, img_path in enumerate(self.imgs):
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                self.masks.append(json_path)
            else:
                self.imgs.remove(img_path)  # del img if have not maskfile
                print(f'{str(json_path)} does not exist!')

        self.input_size = (input_width, input_height)

        # convert str names to class values on masks
        self.classes = classes
        self.class_values = [idx + 1 for idx, c in enumerate(self.classes)]

        self.augmentation = albu.Compose(augmentation) if augmentation else augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def draw_mask(img, mask, threshhold=0.1, mode='cv2'):
        colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
        ]

        # empty_image = np.zeros(img.shape)
        img_mask = img.copy()
        chs = []
        for ch in range(mask.shape[-1]):
            bool_class_mask = np.where(mask[:, :, ch] > threshhold)
            if len(bool_class_mask[0]) > 0:
                chs.append(ch)
                # empty_image[bool_class_mask] = colors[ch]
                img_mask[bool_class_mask] = colors[ch]
        img_mask = cv2.cvtColor(img_mask.astype(np.uint8), cv2.COLOR_BGR2RGB)
        # empty_image = cv2.cvtColor(empty_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        plt_show_img(img_mask, title=f'detected classes: {chs}', mode=mode)
        # plt_show_img(empty_image)

    def __getitem__(self, item):
        img_path, mask_path = self.imgs[item], self.masks[item]
        assert img_path.stem == mask_path.stem, f'img_path.stem!=mask_path.stem ("{img_path.stem}" and "{mask_path.stem}")'

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.shape[:2] != self.input_size[::-1]:
            image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_NEAREST)

        ''' # img mask
        mask_path_img = img_path.parent.parent / 'annot' / img_path.name  # TODO: del!
        mask = cv2.imread(str(mask_path_img), 0)  # читаем только 1 канал. Не хорошо!
        if mask.shape != self.input_size[::-1]:
            mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        # /img mask '''

        # json mask
        mask = self.make_mask_from_json(mask_path)
        # /json mask

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # self.draw_mask(image, mask, mode='plt')  # for augmentation demonstration
        # exit()

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask

    def make_mask_from_json(self, json_path):
        with open(json_path, 'r') as label_json:
            json_txt = json.load(label_json)
        orig_h, orig_w = json_txt["imageHeight"], json_txt["imageWidth"]
        labels = json_txt["shapes"]
        mask = np.zeros((orig_h, orig_w, len(self.classes)), dtype=np.uint8)

        for l in labels:
            class_index = self.classes.index(l["label"])
            points = l["points"]

            if l["shape_type"] in ["polygon", "linestrip"]:
                color = [0] * len(self.classes)
                color[class_index] = 1

                contour = [np.array(points, dtype=np.int32)]
                cv2.drawContours(
                    image=mask,
                    contours=[contour[0]],
                    contourIdx=0,
                    color=color,
                    thickness=-1)
            elif l["shape_type"] in ["rectangle"]:
                cv2.rectangle(mask,
                              (int(points[0][0]), int(points[0][1])),
                              (int(points[1][0]), int(points[1][1])),
                              color, -1)
        if mask.shape[:2] != self.input_size[::-1]:
            mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)
        return mask


class SmokeModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, activation,
                 loss_fn, out_classes,
                 in_channels=3, start_learning_rate=1e-3,
                 **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, activation=activation, in_channels=in_channels, classes=out_classes,
            **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)

        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, in_channels, 1, 1))
        self.register_buffer("std", torch.tensor(params["std"]).view(1, in_channels, 1, 1))

        self.loss_fn = loss_fn
        self.start_learning_rate = start_learning_rate

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std  # need this?
        # TODO: compare blob with opencv blob
        pred = self.model(image)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.start_learning_rate,
            weight_decay=1e-3,
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.5,
        #     patience=10, cooldown=5,
        #     min_lr=1e-7, eps=1e-7,
        #     verbose=True,
        # )

        major_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        # scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=major_scheduler)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": 'loss/validation'}
        return {"optimizer": optimizer, "lr_scheduler": major_scheduler, "monitor": 'loss/validation'}

    def _shared_step(self, batch, stage):
        imgs, gts = batch
        preds = self.forward(imgs)

        # - **y_pred** - torch.Tensor of shape NxCxHxW
        # - **y_true** - torch.Tensor of shape NxHxW or Nx1xHxW


        ''' list of losses
        total_loss = 0
        for loss in self.loss_fn:
            total_loss += loss(preds, gts)
        # '''

        total_loss = self.loss_fn(preds, gts)


        self.log(f'loss/{stage}', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': total_loss, 'preds': preds}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage='validation')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, stage='test')


class MetricSMPCallback(Callback):
    def __init__(self, metrics, mode='multiclass', reduction='micro-imagewise', n_img_valid_check_per_epoch=0) -> None:
        # maybe binary because each chanel is binary?
        # from torchmetrics.classification import BinaryJaccardIndex ?
        # https://torchmetrics.readthedocs.io/en/stable/classification/jaccard_index.html#jaccard-index
        self.metrics = metrics
        self.mode = mode
        self.reduction = reduction
        self.n_img_valid_check_per_epoch = n_img_valid_check_per_epoch

    def _get_metrics(self, preds, gts):
        metric_results = {k: [] for k in self.metrics}
        tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), gts.long(), mode=self.mode)
        # TODO: добавить возможность подсчета метрик по каналам, т.е. по классам (ignore_index?)
        for m_name, metric in self.metrics.items():
            per_image_metric = metric(tp, fp, fn, tn, reduction=self.reduction)
            metric_results[m_name] = round(per_image_metric.item(), 4)
        return metric_results

    @staticmethod
    def _get_full_color(shape, color):
        channels = []
        for ch in range(len(color)):
            ch_color = torch.full(shape, color[ch])
            channels.append(ch_color)
        return torch.stack(channels[::-1], dim=0)

    @staticmethod
    @torch.no_grad()
    def draw_tensor_masks(img_t, gt_mask_t, pred_mask_t, concat_dim=1):  # need for speed
        colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
        ]
        device = img_t.get_device()

        # [-1:1]->[0:255] RGB->BGR
        img_mask_gt = (img_t[[2, 1, 0], :] * 0.5 + 0.5) * 255  # shape = [3, h, w]
        img_shape = img_mask_gt.shape[1:]  # [h,w]
        img_mask_pred = img_mask_gt.clone()
        for ch_idx, (ch_gt, ch_pred) in enumerate(zip(gt_mask_t, pred_mask_t)):
            confidences_gt = torch.stack((ch_gt, ch_gt, ch_gt), dim=0)  # shape = [3, h, w]
            confidences_pred = torch.stack((ch_pred, ch_pred, ch_pred), dim=0)  # shape = [3, h, w]

            full_color = MetricSMPCallback._get_full_color(img_shape, colors[ch_idx]).to(device)

            img_mask_gt = img_mask_gt * (1.0 - confidences_gt) + full_color * confidences_gt
            img_mask_pred = img_mask_pred * (1.0 - confidences_pred) + full_color * confidences_pred

        pred_gt_concat = torch.cat([img_mask_gt, img_mask_pred], dim=concat_dim)
        pred_gt_concat = torch.permute(pred_gt_concat, (1, 2, 0))
        return pred_gt_concat.cpu().detach().numpy()

    def _on_shared_batch_end(self, trainer, outputs, batch, batch_idx, stage) -> None:
        imgs, gts = batch
        preds = outputs['preds']
        metrics = self._get_metrics(preds=preds, gts=gts)
        for m_name, m_val in metrics.items():
            trainer.model.log(f'{m_name}/{stage}', m_val, on_step=False, on_epoch=True)

        if stage in ['validation'] and batch_idx in self.batches_check:
        # if stage in ['train'] and batch_idx in [0]:
            img_for_show = self.draw_tensor_masks(imgs[0], gts[0], preds[0])
            title = f'epoch={trainer.current_epoch} batch_idx={batch_idx}\n{metrics}'
            plt_show_img(img_for_show, title=title, mode='plt')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._on_shared_batch_end(trainer, outputs, batch, batch_idx, stage='train')

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.batches_check = random.sample(
            range(0, trainer.val_check_batch), min(self.n_img_valid_check_per_epoch, trainer.val_check_batch)
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._on_shared_batch_end(trainer, outputs, batch, batch_idx, stage='validation')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._on_shared_batch_end(trainer, outputs, batch, batch_idx, stage='test')
