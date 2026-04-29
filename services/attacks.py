import cv2
import numpy as np
import math

class Attack:
    @staticmethod
    def gaussian_noise(img: np.ndarray, std=10):
        img = img.copy()
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        noisy_img = cv2.add(img.astype(np.float32), noise)
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    @staticmethod
    def salt_pepper_noise(img: np.ndarray, prob=0.01):
        img = img.copy()
        rnd = np.random.rand(*img.shape[:2])
        if img.ndim == 3:
            img[rnd < prob/2] = [0, 0, 0]
            img[rnd > 1 - prob/2] = [255, 255, 255]
        else:
            img[rnd < prob/2] = 0
            img[rnd > 1 - prob/2] = 255
        return img

    @staticmethod
    def median_filter(img: np.ndarray, kernel_size=3):
        return cv2.medianBlur(img, kernel_size)

    @staticmethod
    def gaussian_blur(img: np.ndarray, kernel_size=5, sigma=1.0):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    @staticmethod
    def jpeg_compression(img: np.ndarray, quality=70):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def rotation(img: np.ndarray, angle=5):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def scaling(img: np.ndarray, scale_factor=0.8):
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def cropping(img: np.ndarray, crop_ratio=0.1):
        img_copy = img.copy()
        h, w = img_copy.shape[:2]
        crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
        if img_copy.ndim == 3:
            img_copy[:crop_h, :] = 0
            img_copy[-crop_h:, :] = 0
            img_copy[:, :crop_w] = 0
            img_copy[:, -crop_w:] = 0
        else:
            img_copy[:crop_h, :] = 0
            img_copy[-crop_h:, :] = 0
            img_copy[:, :crop_w] = 0
            img_copy[:, -crop_w:] = 0
        return img_copy

    @staticmethod
    def blur(img: np.ndarray):
        return cv2.blur(img, (2, 2))

    @staticmethod
    def rotate180(img: np.ndarray):
        return cv2.rotate(img, cv2.ROTATE_180)

    @staticmethod
    def rotate90(img: np.ndarray):
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    @staticmethod
    def chop5(img: np.ndarray):
        return Attack.cropping(img, 0.05)

    @staticmethod
    def saltnoise(img: np.ndarray):
        return Attack.salt_pepper_noise(img, 0.01)

    @staticmethod
    def brighter10(img: np.ndarray):
        return np.clip(img.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)

    @staticmethod
    def darker10(img: np.ndarray):
        return np.clip(img.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)

    @staticmethod
    def largersize(img: np.ndarray):
        return Attack.scaling(img, 1.5)

    @staticmethod
    def smallersize(img: np.ndarray):
        return Attack.scaling(img, 0.5)
