import glob
import os
import random

import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from skimage import morphology
from torch.utils.data import Dataset


def data_augmentation(image, mode):
    """
    Performs data augmentation of the input PIL image
    Input:
        image: a PIL image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    Output:
        Augmented PIL image
    """
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = ImageOps.flip(image)
    elif mode == 2:
        # rotate counterclockwise 90 degrees
        out = image.rotate(90, expand=True)
    elif mode == 3:
        # rotate 90 degrees and flip up and down
        out = image.rotate(90, expand=True)
        out = ImageOps.flip(out)
    elif mode == 4:
        # rotate 180 degrees
        out = image.rotate(180, expand=True)
    elif mode == 5:
        # rotate 180 degrees and flip
        out = image.rotate(180, expand=True)
        out = ImageOps.flip(out)
    elif mode == 6:
        # rotate 270 degrees
        out = image.rotate(270, expand=True)
    elif mode == 7:
        # rotate 270 degrees and flip
        out = image.rotate(270, expand=True)
        out = ImageOps.flip(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out


class ImageDataset(Dataset):
    def __init__(self, root1, root2=None, transforms_=None, mode="train", file_set="separate", color_set="RGB"):
        """
        root1: the input pics
        root2: the target pics that we want to preceed
        transforms_: the transform functions of each pic
        mode: results different ML mode for data selection, common options "train", "val"
        file_set: depends on two different resources for input pic:
          "combined" refers to the combination of input and target pic;
          "separate" refers to separated address of input and target pic
        color_set: color set of an image, "RGB" or "L"
        """

        self.file_set = file_set
        self.transform = transforms.Compose(transforms_)
        self.color_set = color_set

        if self.file_set == "combined":
            self.files = sorted(glob.glob(os.path.join(root1, mode) + "/*.*"))

        elif self.file_set == "separate":
            self.files = []
            for pics in sorted(glob.glob(root1 + "/*.*")):
                pic_input = os.path.join(root1, pics.split("/")[-1])
                pic_target = pic_input.replace('input', 'target')
                if os.path.exists(pic_target):
                    self.files.append((pic_input, pic_target))

    def __getitem__(self, index):
        # process the combination of target and input pic
        if self.file_set == "combined":
            img = Image.open(self.files[index % len(
                self.files)]).convert(self.color_set)
            w, h = img.size

            # separate the input and target pic from the pic combination
            img_A = img.crop((0, 0, w / 2, h))
            img_B = img.crop((w / 2, 0, w, h))
        # read two pics separately
        elif self.file_set == "separate":
            (input, target) = self.files[index % len(self.files)]
            # "L" for grayscale images, "RGB" for color images
            img_A = Image.open(target).convert(self.color_set)
            img_B = Image.open(input).convert(self.color_set)

        # reverse a pic for data augmentation
        # if self.color_set == "RGB" and np.random.random() < 0.5:
        #     img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
        #     img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
        # if self.color_set == "L" and np.random.random() < 0.5:
        #     img_A = Image.fromarray(np.array(img_A)[:, ::-1], "L")
        #     img_B = Image.fromarray(np.array(img_B)[:, ::-1], "L")

        # flag_aug = random.randint(0,7)
        # img_A = data_augmentation(img_A, flag_aug)
        # img_B = data_augmentation(img_B, flag_aug)

        sk = Image.fromarray(skeletonize(
            cv2.imread(target, cv2.IMREAD_GRAYSCALE)))
        # sk = data_augmentation(sk, flag_aug)

        # precess image transform functions
        trans = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()])
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        sk = trans(sk)

        return {"A": img_A, "B": img_B, "F": sk, "target_path": target}

    def __len__(self):
        return len(self.files)


def skeletonize(image):
    # improved skeletonization function
    image = np.uint8(image)
    ret, binary = cv2.threshold(
        image, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)
    skeleton = skeleton0.astype(np.uint8) * 255
    skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    return skeleton
