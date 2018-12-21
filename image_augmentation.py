from PIL import Image
import os
import numpy as np
import cv2
from random import randint, random
import time
from collections import deque
from imgaug import augmenters as iaa
import imgaug as ia


# 50% chance to do the augmentation
def sometimes(aug, chance=0.5):
    return iaa.Sometimes(chance, aug)


class ImageMixer:
    def __init__(self, data_dir, image_size=128, augmentations=50,
                 crop_percent=(-0.07, 0.1), affine_scale=(0.8, 1.2),
                 hue_range=(0, 20), translate_percent=(-0.1, 0.1),
                 rotation=(-10, 10)):
        """

        Args:
            data_dir:
            image_size:
            jpeg_pics:
            different_crops:
            crop_diff_w:
            crop_diff_h:
            keep_close_aspect_ratio: If None will randomly keep aspect ratio
        """
        self.data_dir = data_dir
        self._data = []
        self.image_size = image_size
        self.num_augmentations = augmentations

        self.seq = iaa.Sequential(
            [
                iaa.OneOf([
                    # Crop images to -7% to 10% of their width/height
                    sometimes(iaa.CropAndPad(
                        percent=crop_percent,
                    ), chance=0.2),

                    # Scale image between 80% to 120% of original size
                    # Translate the picture -10% to 10% on both axes
                    sometimes(iaa.Affine(
                        scale=affine_scale,
                        translate_percent=translate_percent,
                    ), chance=0.4)
                ]),

                # Rotate and shear image
                sometimes(iaa.Affine(
                    rotate=rotation,
                    shear=(-3, 3),
                    mode=ia.ALL
                ), chance=0.2),

                # Changes gamma contrast
                sometimes(iaa.GammaContrast(
                    gamma=(0.8, 1.3)
                ), chance=0.3),

                # Change to HSV and add hue then transfer back to RGB
                sometimes([
                    iaa.ChangeColorspace(from_colorspace="RGB",
                                         to_colorspace="HSV"),
                    iaa.WithChannels(0, iaa.Add(hue_range)),
                    iaa.ChangeColorspace(from_colorspace="HSV",
                                         to_colorspace="RGB")
                ], chance=0.2),

                # Add one type of blur
                sometimes(
                    iaa.OneOf([
                        iaa.GaussianBlur(sigma=(0.1, 2)),
                        iaa.AverageBlur(k=(1, 6)),
                        iaa.MedianBlur(k=(1, 7)),
                        iaa.BilateralBlur(d=(1, 7), sigma_color=250,
                                          sigma_space=250)
                    ]), chance=0.4),

                sometimes(
                    iaa.Sharpen(alpha=(0, 0.4)),
                    chance=0.4
                )
            ],
            random_order=True
        )

    def mix_it_up(self):
        folder = os.listdir(self.data_dir)
        print()
        idx = 1
        d = deque(maxlen=100)
        for f in folder:
            t = time.time()
            f = os.path.join(self.data_dir, f)
            if not os.path.isfile(f):
                self.mix_image(f)

            d.append(time.time() - t)
            eta = (sum(d) / len(d)) * (len(folder) - idx)
            print('\r' + ' ' * 200, end='')
            print(f'\r{idx}/{len(folder)} ETA {eta:.02f}s', end='')
            idx += 1

    def mix_image(self, folder):
        file = os.listdir(folder)[-1]

        file = os.path.join(folder, file)
        im = Image.open(file)
        bg = Image.new('RGBA', im.size, (0,0,0,255))
        bg.paste(im, mask=im.convert('RGBA'))
        im = np.array(bg)[:,:,:3]  # Ignore alpha

        for i in range(self.num_augmentations):
            p = os.path.join(folder, f'{i}.png')
            Image.fromarray(self.seq.augment_image(im), 'RGB').resize((self.image_size, self.image_size), Image.LINEAR).save(p)


def read_data(data_dir):
    classes = []
    data = []
    for f in os.listdir(data_dir):
        classname = f
        f = os.path.join(data_dir, f)
        if os.path.isfile(f):
            continue

        classes.append(classname)
        for file in os.listdir(f):
            if file.endswith('.jpg'):
                dat = cv2.imread(os.path.join(f, file))
                dat = np.multiply(dat, 1.0 / 255.0)
                data.append(dat)

    return data, classes
