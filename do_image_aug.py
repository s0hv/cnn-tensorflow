import image_augmentation

train_path = 'data'

image_augmentation.ImageMixer(train_path, augmentations=100, affine_scale=(0.8, 1.1),
                              hue_range=(0, 5), rotation=(-5, 5),
                              translate_percent=(-0.1, 0.1),
                              crop_percent=(-0.05, 0.05)).mix_it_up()
