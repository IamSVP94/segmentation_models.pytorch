import json
import random
from pathlib import Path

import albumentations as albu
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from torch.utils.data import Dataset as BaseDataset

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.modules import Activation
from svp_customs.utills import glob_search, plt_show_img, get_random_colors, max_show_img_size_reshape


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
                 loss_fn, classes,
                 in_channels=3, start_learning_rate=1e-3,
                 **kwargs):
        super().__init__()
        self.classes = classes
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, activation=activation, in_channels=in_channels, classes=len(self.classes),
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
            # weight_decay=1e-3,
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
    def __init__(self,
                 metrics,
                 classes_separately: bool = False,
                 colors=None,
                 mode='multiclass',
                 reduction='micro-imagewise',
                 activation='identity',  # 'identity' just return input
                 threshold=None,
                 n_img_check_per_epoch_validation: int = 0,
                 n_img_check_per_epoch_train: int = 0,
                 n_img_check_per_epoch_save: bool = False,
                 save_img: bool = False,
                 log: bool = False,
                 ) -> None:
        self.metrics = metrics
        self.classes_separately = classes_separately
        self.colors = get_random_colors(80) if colors is None else colors

        self.mode = mode
        self.reduction = reduction

        self.threshold = threshold
        self.activation = Activation(activation)

        self.n_img_check_per_epoch_validation = n_img_check_per_epoch_validation
        self.n_img_check_per_epoch_train = n_img_check_per_epoch_train
        self.n_img_check_per_epoch_save = n_img_check_per_epoch_save
        self.save_img = save_img
        self.log = log

    def _get_metrics(self, preds, gts, trainer):
        metric_results = {k: dict() for k in self.metrics}

        preds = self._get_preds_after_threshold(preds, self.threshold) if self.threshold else preds

        tp, fp, fn, tn = smp.metrics.get_stats(
            output=preds.long(),
            target=gts.long(),
            mode=self.mode,
        )

        if self.classes_separately:
            for m_name, metric in self.metrics.items():
                metric_results[m_name] = {}
                for cl_idx, cl_name in enumerate(trainer.model.classes):
                    cl_tp = tp[:, cl_idx].unsqueeze(-1)
                    cl_fp = fp[:, cl_idx].unsqueeze(-1)
                    cl_fn = fn[:, cl_idx].unsqueeze(-1)
                    cl_tn = tn[:, cl_idx].unsqueeze(-1)

                    per_image_metric = metric(cl_tp, cl_fp, cl_fn, cl_tn, reduction=self.reduction)
                    metric_results[m_name][cl_name] = round(per_image_metric.item(), 4)
                else:
                    per_image_metric = metric(tp, fp, fn, tn, reduction=self.reduction)
                    metric_results[m_name]['together'] = round(per_image_metric.item(), 4)
        return metric_results

    def _get_metrics_old(self, preds, gts):
        metric_results = {k: [] for k in self.metrics}
        num_classes = preds.shape[1]

        preds = torch.where(preds >= self.threshold, 1, 0) if self.threshold else preds

        tp, fp, fn, tn = smp.metrics.get_stats(
            output=preds.long(),
            target=gts.long(),
            mode=self.mode,
            # threshold=threshold,  # bug with threshold and *.long() format
            num_classes=num_classes,
            # ignore_index=self.ignore_index,
        )
        for m_name, metric in self.metrics.items():
            per_image_metric = metric(tp, fp, fn, tn, reduction=self.reduction)
            metric_results[m_name] = round(per_image_metric.item(), 4)
        return metric_results

    @staticmethod
    def _get_full_color(shape, color, rgb2bgr=False):
        channels = []
        for ch in range(len(color)):
            ch_color = torch.full(shape, color[ch])
            channels.append(ch_color)
        if rgb2bgr:
            channels = channels[::-1]
        return torch.stack(channels, dim=0)

    @staticmethod
    def _get_preds_after_threshold(preds, threshold):
        preds_thr = []
        for cl in range(preds.shape[-3]):  # skip h,w from tail
            if len(preds.shape) == 3:
                ch_preds = torch.where(preds[cl, :, :] >= threshold[cl], 1, 0)
            elif len(preds.shape) == 4:
                ch_preds = torch.where(preds[:, cl, :, :] >= threshold[cl], 1, 0)
            preds_thr.append(ch_preds)
        return torch.stack(preds_thr, dim=-3)

    @torch.no_grad()
    def draw_tensor_masks(self, img_t, gt_mask_t, pred_mask_t, to_numpy=False):  # need for speed
        device = img_t.get_device()

        pred_mask_t_threshold = self._get_preds_after_threshold(
            pred_mask_t, self.threshold) if self.threshold else pred_mask_t

        # [-1:1]->[0:255] RGB->BGR
        img_t = (img_t * 0.5 + 0.5) * 255  # shape = [3, h, w]
        img_shape = img_t.shape[1:]  # [h,w]

        img_mask_gt, img_mask_pred, img_mask_pred_thr = img_t.clone(), img_t.clone(), img_t.clone()
        for ch_idx, (ch_gt, ch_pred, ch_pred_thr) in enumerate(zip(gt_mask_t, pred_mask_t, pred_mask_t_threshold)):
            # shape = [3, h, w]
            confidences_gt = torch.stack((ch_gt, ch_gt, ch_gt), dim=0)
            confidences_pred = torch.stack((ch_pred, ch_pred, ch_pred), dim=0)
            confidences_pred_thr = torch.stack((ch_pred_thr, ch_pred_thr, ch_pred_thr), dim=0)

            full_color = MetricSMPCallback._get_full_color(img_shape, self.colors[ch_idx]).to(device)

            img_mask_gt = img_mask_gt * (1.0 - confidences_gt) + full_color * confidences_gt
            img_mask_pred = img_mask_pred * (1.0 - confidences_pred) + full_color * confidences_pred
            img_mask_pred_thr = img_mask_pred_thr * (1.0 - confidences_pred_thr) + full_color * confidences_pred_thr

        orig_gt_concat = torch.cat([img_t, img_mask_gt], dim=2)  # concat by w
        pred_pred_th_concat = torch.cat([img_mask_pred, img_mask_pred_thr], dim=2)  # concat by w

        final_img = torch.cat([orig_gt_concat, pred_pred_th_concat], dim=1)  # concat by h
        final_img = torch.permute(final_img, (1, 2, 0))
        if to_numpy:
            final_img = final_img.cpu().detach().numpy()
        return final_img

    @staticmethod
    def _cv2_add_title(img, title,
                       font=cv2.FONT_HERSHEY_COMPLEX,
                       font_scale=1, thickness=2,
                       where='top', color=(0, 0, 0),
                       ):
        lines_w = lines_h = 0
        lines = []
        for i, line in enumerate(title.split('\n')):
            lines.append(line)
            (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, thickness)
            lines_w += text_w
            lines_h += 2 * text_h

        if where == 'top':
            text_pos_x, text_pos_y = 5, int(lines_h / len(lines))
            top = lines_h + text_h
            bottom = left = right = 0
        if where == 'bottom':
            pass
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (255, 255, 255))

        for line in lines:
            cv2.putText(img, line, (text_pos_x, text_pos_y), font, font_scale, color, thickness)
            if where == 'top':
                text_pos_y += int(lines_h / len(lines))
        return img

    @torch.no_grad()
    def _on_shared_batch_end(self, trainer, outputs, batch, batch_idx, stage) -> None:
        imgs, gts = batch
        preds = self.activation(outputs['preds'])

        loss = round(outputs['loss'].item(), 5)

        metrics = self._get_metrics(preds=preds, gts=gts, trainer=trainer)

        for m_name, m_vals in metrics.items():
            for cl_name, m_val in m_vals.items():
                trainer.model.log(f'{m_name}/{stage}/{cl_name}', m_val, on_step=False, on_epoch=True)

        if stage in ['validation'] and batch_idx in self.batches_check_validation or \
                stage in ['train'] and batch_idx in self.batches_check_train:
            save_path = None
            if self.n_img_check_per_epoch_save:
                log_path = Path(trainer.model.logger.experiment.get_logdir()) / 'imgs' / stage
                save_path = log_path / f'epoch={trainer.current_epoch}_batch_idx={batch_idx}.jpg'
                save_path.parent.mkdir(parents=True, exist_ok=True)

            title = f'{stage}: epoch={trainer.current_epoch} batch_idx={batch_idx} (loss={loss})\n'
            title += f'threshold={self.threshold}\n'
            for m_name, cl_vals in metrics.items():
                for cl_name, m_cl_val in cl_vals.items():
                    title += f'{m_name}: {cl_name} - {m_cl_val}\n'

            img_for_show = self.draw_tensor_masks(imgs[0], gts[0], preds[0], to_numpy=True)
            img_for_show = max_show_img_size_reshape(img_for_show, max_show_img_size=(1400, 1400)).astype(np.uint8)
            img_for_show = self._cv2_add_title(img_for_show, title)

            if self.log:
                trainer.model.logger.experiment.add_image(
                    tag=f'{stage}/{batch_idx}',
                    img_tensor=img_for_show,
                    global_step=trainer.current_epoch,
                    dataformats='HWC'
                )
            if save_path is not None:
                if self.save_img:
                    cv2.imwrite(str(save_path), cv2.cvtColor(img_for_show, cv2.COLOR_BGR2RGB))

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if trainer.current_epoch == 100:  # if 0 epoch not need to save
            self.batches_check_train = []
        else:
            self.batches_check_train = random.sample(
                range(0, trainer.val_check_batch),
                min(self.n_img_check_per_epoch_train, trainer.val_check_batch)
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._on_shared_batch_end(trainer, outputs, batch, batch_idx, stage='train')

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.batches_check_validation = random.sample(
            range(0, trainer.val_check_batch),
            min(self.n_img_check_per_epoch_validation, trainer.val_check_batch)
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._on_shared_batch_end(trainer, outputs, batch, batch_idx, stage='validation')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._on_shared_batch_end(trainer, outputs, batch, batch_idx, stage='test')
