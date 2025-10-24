from .config import DIMENSIONS, MIXED_PRECISION

import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from pycocotools.coco import COCO


class COCOSegmentationGenerator(Sequence):
    def __init__(self, images_dir, annotation_file, target_class_id=2,
                 batch_size=4, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.images_dir = images_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.target_class_id = target_class_id
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        # Индексы картинок для батча
        batch_ids = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        X, Y = self.__data_generation(batch_ids)
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        batch_size = len(batch_ids)

        if MIXED_PRECISION:
            # Картинки в float16 для mixed precision
            X = np.zeros((batch_size, DIMENSIONS[0], DIMENSIONS[1], 3), dtype=np.float16)
        else:
            X = np.zeros((batch_size, DIMENSIONS[0], DIMENSIONS[1], 3), dtype=np.float32)
        # Маски в float32 для корректной работы loss/metrics
        Y = np.zeros((batch_size, DIMENSIONS[0], DIMENSIONS[1], 1), dtype=np.float32)

        for i, idx in enumerate(batch_ids):
            img_id = self.image_ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.images_dir, img_info["file_name"])

            # --- Загружаем изображение ---
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (DIMENSIONS[1], DIMENSIONS[0]))
            if MIXED_PRECISION:
                image = image.astype(np.float32) / 255.0  # нормализация
                image = image.astype(np.float16)          # mixed precision
            else:
                image = image / 255.0  # нормализация

            # --- Загружаем аннотации ---
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
            for ann in anns:
                if ann["category_id"] == self.target_class_id:
                    mask_part = self.coco.annToMask(ann)
                    mask = np.maximum(mask, mask_part)

            mask = cv2.resize(mask, (DIMENSIONS[1], DIMENSIONS[0]), interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1)  # (H,W,1)
            mask = mask.astype(np.float32)

            X[i] = image
            Y[i] = mask

        return X, Y
