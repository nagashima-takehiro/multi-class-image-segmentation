import albumentations as albu
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

CLASSES = [
    'background',
    'asufalt',
    'bigrock',
    'glass',
    'renga',
    'rockmid',
    'rocksmall',
]

COLOR_PALETTE = [
    [0, 0, 0],
    [255, 165, 0],
    [255, 255, 0],
    [0, 128, 0],
    [0, 255, 255],
    [0, 0, 255],
    [128, 0, 128],
]
COLOR_PALETTE = np.array(COLOR_PALETTE).reshape(-1).tolist()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class MaterialDataset(torch.utils.data.Dataset):
    CLASSES = CLASSES

    def __init__(
        self,
        images_path,
        masks_path,
        classes,
        augmentation=None,
        preprocessing=None,
    ):
        self.images_path = images_path
        self.masks_path = masks_path
        self.classes = classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = Image.open(self.images_path[i])
        image = crop_to_square(image)
        #  image = image.resize((128,128), Image.ANTIALIAS)
        image = image.resize((512,512), Image.LANCZOS)
        image = np.asarray(image)
        masks = Image.open(self.masks_path[i])
        masks = crop_to_square(masks)
        #  masks = masks.resize((128,128), Image.ANTIALIAS)
        masks = masks.resize((512,512), Image.LANCZOS)
        masks = np.asarray(masks)
        # masks = np.where(masks == 255, 21, masks)
        cls_idx = [self.CLASSES.index(cls) for cls in self.classes]
        masks = [(masks == idx) for idx in cls_idx]
        mask = np.stack(masks, axis=-1).astype("float")
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        return image, mask

    def __len__(self):
        return len(self.images_path)

def get_augmentation(phase):
    if phase == 'train':
        train_transform = [
            # albu.HorizontalFlip(p=0.5),
            albu.Flip(),
            albu.ShiftScaleRotate(scale_limit=(-0.8, 0.0), p=1, border_mode=0),
            albu.PadIfNeeded(min_height=1024, min_width=1024, always_apply=True, border_mode=0),
            # albu.RandomCrop(height=320, width=320, always_apply=True),
            # albu.RandomCrop(height=20, width=20, always_apply=True),
            # albu.IAAAdditiveGaussianNoise(p=0.2),
            # albu.IAAPerspective(p=0.5),
            # albu.OneOf(
            #     [
            #         albu.CLAHE(p=1),
            #         albu.RandomBrightness(p=1),
            #         albu.RandomGamma(p=1),
            #     ],
            #     p=0.9,
            # ),
            # albu.OneOf(
            #     [
            #         albu.IAASharpen(p=1),
            #         albu.Blur(blur_limit=3, p=1),
            #         albu.MotionBlur(blur_limit=3, p=1),
            #     ],
            #     p=0.9,
            # ),
            # albu.OneOf(
            #     [
            #         albu.RandomContrast(p=1),
            #         albu.HueSaturationValue(p=1),
            #     ],
            #     p=0.9,
            # ),
        ]
        return albu.Compose(train_transform)
    elif phase == 'valid':
        return None

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def crop_to_square(image):
    size = min(image.size)
    left, upper = (image.width - size) // 2, (image.height - size) // 2
    right, bottom = (image.width + size) // 2, (image.height + size) // 2
    return image.crop((left, upper, right, bottom))

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def get_list_img(path: str, ext: str = 'jpg') -> list:
    return sorted(glob.glob(f'{path}/*.{ext}'))

