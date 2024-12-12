import math
import random

import cv2
import numpy as np
import scipy
import torch
from PIL import Image
from sklearn import metrics
from torch.nn import functional


def setup_seed():
    """Sets the random seed for reproducibility."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_multiprocessing():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)
    num_threads = '1'
    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = num_threads

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = num_threads


def fitness(x):
    # fitness as a weighted combination of metrics
    w = [1.0, 1.0, 1.0, 1.0]  # weights for metrics
    return np.sum(x * w, 0)


def compute_metrics(outputs, targets):
    roc_auc = metrics.roc_auc_score(targets, outputs)
    precision, recall, _ = metrics.precision_recall_curve(targets, outputs)
    auc = metrics.auc(recall, precision)
    f1 = metrics.f1_score(targets, (outputs > 0.1).astype('int32'))
    accuracy = metrics.accuracy_score(targets, (outputs > 0.1).astype('int32'))

    return roc_auc, auc, f1, accuracy


def generate_anomaly_mask(args, scores, device):
    """Generates anomaly masks by applying Gaussian smoothing to anomaly scores."""
    with torch.no_grad():
        if isinstance(scores, np.ndarray):
            scores = torch.from_numpy(scores)
        scores = scores.to(device)
        scores = scores.unsqueeze(1)
        scores = functional.interpolate(scores,
                                        size=(args.input_size, args.input_size),
                                        mode="bilinear", align_corners=False)
        scores = scores.squeeze(1).cpu().numpy()

    return [scipy.ndimage.gaussian_filter(score, sigma=4) for score in scores]


def resample():
    return random.choice((0, 1, 2, 3, 4))


class ZeroPadding:
    """Pads an image to a square shape."""
    def __call__(self, image):
        w, h = image.size
        if w == h:
            return image
        elif w > h:
            result = Image.new(image.mode, (w, w), (0, 0, 0))
            result.paste(image, (0, (w - h) // 2))
            return result
        else:
            result = Image.new(image.mode, (h, h), (0, 0, 0))
            result.paste(image, ((h - w) // 2, 0))
            return result


class RandomAugment:
    """Applies random augmentations to an image, including color space adjustments and geometric transformations."""
    def __init__(self, p=0.5):
        self.p = p
        self.angle = (30, 30)
        self.scale = (0.05, 0.05)
        self.shear = (0.05, 0.05)
        self.translate = (0.05, 0.05)

    def __call__(self, image):
        if random.random() < self.p:
            image = np.array(image)
            # HSV color-space augmentation
            x = np.arange(0, 256, dtype=np.int32)
            hsv = np.random.uniform(-1, 1, 3) * [0.4, 0.4, 0.4] + 1
            h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))

            lut_h = ((x * hsv[0]) % 180).astype('uint8')
            lut_s = np.clip(x * hsv[1], 0, 255).astype('uint8')
            lut_v = np.clip(x * hsv[2], 0, 255).astype('uint8')

            h = cv2.LUT(h, lut_h)
            s = cv2.LUT(s, lut_s)
            v = cv2.LUT(v, lut_v)

            image_hsv = cv2.merge((h, s, v)).astype('uint8')
            cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB, dst=image)
            h, w = image.shape[:2]
            # Center
            center = np.eye(3)
            center[0, 2] = -w / 2  # x translation (pixels)
            center[1, 2] = -h / 2  # y translation (pixels)

            # Perspective
            perspective = np.eye(3)

            # Rotation and Scale
            rotation = np.eye(3)
            a = random.uniform(-self.angle[0], self.angle[1])
            s = random.uniform(1 - self.scale[0], 1 + self.scale[1])
            rotation[:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=a, scale=s)

            # Shear
            shear = np.eye(3)
            shear[0, 1] = math.tan(random.uniform(-self.shear[0], self.shear[1]) * math.pi / 180)
            shear[1, 0] = math.tan(random.uniform(-self.shear[0], self.shear[1]) * math.pi / 180)

            # Translation
            translation = np.eye(3)
            translation[0, 2] = random.uniform(0.5 - self.translate[0], 0.5 + self.translate[1]) * w
            translation[1, 2] = random.uniform(0.5 - self.translate[0], 0.5 + self.translate[1]) * h

            # Combined rotation matrix, order of operations (right to left) is IMPORTANT
            matrix = translation @ shear @ rotation @ perspective @ center

            image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), flags=resample())  # affine
            image = Image.fromarray(image)
        return image


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


class ComputeLoss:
    def __init__(self, device):
        super().__init__()
        self.device = device

    def __call__(self, features, model_g, model_d):
        true_feats = model_g(features)

        indices = torch.randint(0, 1, torch.Size([true_feats.shape[0]]))
        one_hot = functional.one_hot(indices, num_classes=1)  # (N, K)
        # Create noisy (fake) features
        noise = torch.stack([torch.normal(0, 0.015, true_feats.shape)], dim=1)  # (N, K, C)
        noise = (noise.to(self.device) * one_hot.to(self.device).unsqueeze(-1)).sum(1)
        fake_feats = true_feats + noise
        
        # Discriminator outputs for true and fake features
        scores = model_d(torch.cat([true_feats, fake_feats]))
        true_scores = scores[:len(true_feats)]
        fake_scores = scores[len(fake_feats):]

        # Calculate proportions for monitoring
        th = 0.5
        p_true = (true_scores.detach() >= th).sum() / len(true_scores)
        p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
        
        # Calculate loss based on discriminator predictions
        true_loss = torch.clip(-true_scores + th, min=0)
        fake_loss = torch.clip(fake_scores + th, min=0)
        loss = true_loss.mean() + fake_loss.mean()

        return loss, p_true, p_fake
