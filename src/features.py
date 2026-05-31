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
    names = []

    for channel_name, c in [("red", 0), ("green", 1), ("blue", 2)]:
        hist, _ = np.histogram(img[:, :, c], bins=bins, range=(0, 1), density=True)
        feats.extend(hist)
        names.extend([f"{channel_name}_hist_bin_{i}" for i in range(bins)])

    return np.array(feats), names


def intensity_features(img):
    gray = color.rgb2gray(img)

    feats = np.array([
        gray.mean(),
        gray.std(),
        gray.min(),
        gray.max(),
        np.percentile(gray, 25),
        np.percentile(gray, 50),
        np.percentile(gray, 75),
    ])

    names = [
        "intensity_mean",
        "intensity_std",
        "intensity_min",
        "intensity_max",
        "intensity_25th_percentile",
        "intensity_median",
        "intensity_75th_percentile",
    ]

    return feats, names


def texture_features(img):
    gray = color.rgb2gray(img)
    gray_uint8 = (gray * 31).astype(np.uint8)

    edges = filters.sobel(gray)

    lbp = feature.local_binary_pattern(gray_uint8, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)

    glcm = feature.graycomatrix(
        gray_uint8,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )

    glcm_props = []
    glcm_names = []

    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        vals = feature.graycoprops(glcm, prop).ravel()
        glcm_props.extend([vals.mean(), vals.std()])
        glcm_names.extend([f"glcm_{prop}_mean", f"glcm_{prop}_std"])

    feats = np.concatenate((
        [edges.mean(), edges.std()],
        lbp_hist,
        np.array(glcm_props),
    ))

    names = (
        ["edge_mean", "edge_std"]
        + [f"lbp_texture_bin_{i}" for i in range(10)]
        + glcm_names
    )

    return feats, names


def fft_features(img):
    gray = color.rgb2gray(img)

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    log_spectrum = np.log1p(spectrum)

    h, w = log_spectrum.shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_radius = radius.max()

    low_mask = radius <= max_radius * 0.20
    mid_mask = (radius > max_radius * 0.20) & (radius <= max_radius * 0.50)
    high_mask = radius > max_radius * 0.50

    low_energy = log_spectrum[low_mask].mean()
    mid_energy = log_spectrum[mid_mask].mean()
    high_energy = log_spectrum[high_mask].mean()

    eps = 1e-8

    feats = np.array([
        log_spectrum.mean(),
        log_spectrum.std(),
        low_energy,
        mid_energy,
        high_energy,
        low_energy / (mid_energy + eps),
        high_energy / (low_energy + eps),
        high_energy / (mid_energy + eps),
    ])

    names = [
        "fft_log_spectrum_mean",
        "fft_log_spectrum_std",
        "fft_low_frequency_energy",
        "fft_mid_frequency_energy",
        "fft_high_frequency_energy",
        "fft_low_mid_ratio",
        "fft_high_low_ratio",
        "fft_high_mid_ratio",
    ]

    return feats, names


def extract_features(path):
    img = load_image(path)

    feature_parts = [
        rgb_histogram_features(img),
        intensity_features(img),
        texture_features(img),
        fft_features(img),
    ]

    values = [part[0] for part in feature_parts]
    return np.concatenate(values)


def get_feature_names():
    dummy = np.zeros((64, 64, 3), dtype=np.float32)

    feature_parts = [
        rgb_histogram_features(dummy),
        intensity_features(dummy),
        texture_features(dummy),
        fft_features(dummy),
    ]

    names = []
    for _, part_names in feature_parts:
        names.extend(part_names)

    return names