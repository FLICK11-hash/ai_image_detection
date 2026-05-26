import numpy as np
from skimage import io, color, transform, filters, feature, util


def load_image(path, image_size=(64, 64)):
    img = io.imread(path)
    if img.ndim == 2:
        img = color.gray2rgb(img)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = transform.resize(img, image_size, anti_aliasing=True)
    return util.img_as_float32(img)


def rgb_histogram_features(img, bins=16):
    feats = []
    for c in range(3):
        hist, _ = np.histogram(img[:, :, c], bins=bins, range=(0, 1), density=True)
        feats.extend(hist)
    return np.array(feats)


def intensity_features(img):
    gray = color.rgb2gray(img)
    return np.array([
        gray.mean(), gray.std(), gray.min(), gray.max(),
        np.percentile(gray, 25), np.percentile(gray, 50), np.percentile(gray, 75)
    ])


def texture_features(img):
    gray = color.rgb2gray(img)
    edges = filters.sobel(gray)
    lbp = feature.local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    return np.concatenate(([edges.mean(), edges.std()], lbp_hist))


def fft_features(img):
    gray = color.rgb2gray(img)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    log_spectrum = np.log1p(spectrum)
    h, w = log_spectrum.shape
    center = log_spectrum[h//4:3*h//4, w//4:3*w//4]
    outer = log_spectrum.copy()
    outer[h//4:3*h//4, w//4:3*w//4] = 0
    return np.array([log_spectrum.mean(), log_spectrum.std(), center.mean(), outer.mean()])


def extract_features(path):
    img = load_image(path)
    return np.concatenate([
        rgb_histogram_features(img),
        intensity_features(img),
        texture_features(img),
        fft_features(img)
    ])
