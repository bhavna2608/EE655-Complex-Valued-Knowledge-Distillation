import torch
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms.functional as F

class CutMix:
    def __init__(self, alpha=1.0):
        """
        CutMix augmentation
        Args:
            alpha (float): Beta distribution parameter (α=1.0 in paper)
        """
        self.alpha = alpha

    def __call__(self, img1, img2, label1, label2):
        """
        Args:
            img1 (PIL Image): First image
            img2 (PIL Image): Second image
            label1: Label for first image
            label2: Label for second image
        Returns:
            mixed_img (PIL Image): Combined image
            labels: Tuple of (label1, label2, lambda)
        """
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get random bounding box coordinates
        W, H = img1.size
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply CutMix
        img1 = img1.copy()
        img2 = img2.copy()
        img1_arr = np.array(img1)
        img2_arr = np.array(img2)
        img1_arr[bby1:bby2, bbx1:bbx2] = img2_arr[bby1:bby2, bbx1:bbx2]
        mixed_img = Image.fromarray(img1_arr)

        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        return mixed_img, (label1, label2, lam)

class AutoAugment:
    def __init__(self):
        """
        AutoAugment policies from the paper
        "AutoAugment: Learning Augmentation Policies from Data"
        """
        self.policies = [
            [('Posterize', 0.4, 8), ('Rotate', 0.6, 9)],
            [('Solarize', 0.6, 5), ('AutoContrast', 0.6, None)],
            [('Equalize', 0.8, None), ('Equalize', 0.6, None)],
            [('Posterize', 0.6, 7), ('Posterize', 0.6, 6)],
            [('Equalize', 0.4, None), ('Solarize', 0.2, 4)]
        ]
    
    def __call__(self, img):
        """
        Apply AutoAugment to an image
        Args:
            img (PIL Image): Input image
        Returns:
            img (PIL Image): Augmented image
        """
        policy = self.policies[np.random.choice(len(self.policies))]
        for op, prob, magnitude in policy:
            if np.random.random() < prob:
                img = self.apply_op(img, op, magnitude)
        return img
    
    def apply_op(self, img, op_name, magnitude):
        """
        Apply a single augmentation operation
        """
        if op_name == 'ShearX':
            img = img.transform(
                img.size, Image.AFFINE, 
                (1, magnitude * np.random.choice([-1, 1]), 0, 0, 1, 0))
        elif op_name == 'ShearY':
            img = img.transform(
                img.size, Image.AFFINE, 
                (1, 0, 0, magnitude * np.random.choice([-1, 1]), 0, 1))
        elif op_name == 'TranslateX':
            img = img.transform(
                img.size, Image.AFFINE, 
                (1, 0, magnitude * img.size[0] * np.random.choice([-1, 1]), 0, 1, 0))
        elif op_name == 'TranslateY':
            img = img.transform(
                img.size, Image.AFFINE, 
                (1, 0, 0, 0, 1, magnitude * img.size[1] * np.random.choice([-1, 1])))
        elif op_name == 'Rotate':
            img = img.rotate(magnitude * np.random.choice([-1, 1]))
        elif op_name == 'AutoContrast':
            img = ImageOps.autocontrast(img)
        elif op_name == 'Invert':
            img = ImageOps.invert(img)
        elif op_name == 'Equalize':
            img = ImageOps.equalize(img)
        elif op_name == 'Solarize':
            img = ImageOps.solarize(img, magnitude)
        elif op_name == 'Posterize':
            img = ImageOps.posterize(img, magnitude)
        elif op_name == 'Contrast':
            img = ImageEnhance.Contrast(img).enhance(1 + magnitude * 0.1)
        elif op_name == 'Color':
            img = ImageEnhance.Color(img).enhance(1 + magnitude * 0.1)
        elif op_name == 'Brightness':
            img = ImageEnhance.Brightness(img).enhance(1 + magnitude * 0.1)
        elif op_name == 'Sharpness':
            img = ImageEnhance.Sharpness(img).enhance(1 + magnitude * 0.1)
        return img