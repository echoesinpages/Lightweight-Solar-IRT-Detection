import numpy as np
import cv2

class AdaptiveMSRCR:
    def __init__(self, scales=(15, 80, 250), alpha=125, gamma=4.6):
        self.scales = scales
        self.alpha = alpha
        self.gamma = gamma

    def single_scale_retinex(self, image, sigma):
        img_float = np.float32(image) + 1.0
        gaussian_blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
        return np.log10(img_float) - np.log10(gaussian_blur)

    def enhance(self, image):
        img_float = np.float32(image) + 1.0
        msr = np.zeros_like(img_float)
        
        for sigma in self.scales:
            msr += self.single_scale_retinex(image, sigma)
        msr = msr / len(self.scales)
        
        channel_sum = np.sum(img_float, axis=2, keepdims=True)
        crf = self.alpha * np.log10(self.gamma * img_float / channel_sum)
        
        msrcr = msr * crf
        
        # Adaptive variance clipping
        for i in range(msrcr.shape[1]):
            channel = msrcr[:, :, i]
            mean, std = np.mean(channel), np.std(channel)
            min_val, max_val = mean - 2 * std, mean + 2 * std
            channel = np.clip(channel, min_val, max_val)
            msrcr[:, :, i] = (channel - min_val) / (max_val - min_val) * 255

        return np.uint8(msrcr)
