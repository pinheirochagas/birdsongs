# %% [markdown]
# # Spectrogram Processing Lab
# Run cells interactively to experiment with denoising, contrast, and color mapping.

# %% Load audio and compute STFT
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, uniform_filter1d
from pathlib import Path

AUDIO_PATH = str(Path(__file__).resolve().parent.parent / "sound" / "uirapuru.mp3")

y, sr = librosa.load(AUDIO_PATH, sr=None, mono=True)
n_fft = 2048
hop_length = 512

S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
S_db_raw = librosa.amplitude_to_db(S, ref=np.max)

fig, ax = plt.subplots(figsize=(14, 5))
img = librosa.display.specshow(S_db_raw, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax)
ax.set_title('Raw spectrogram (dB)')
fig.colorbar(img, ax=ax, format='%+2.0f dB')
plt.tight_layout()
plt.show()

print(f"Duration: {len(y)/sr:.1f}s | Sample rate: {sr} | Shape: {S.shape}")

# %% Denoising method 1: Spectral gating
noise_floor = np.percentile(S, 10, axis=1, keepdims=True)
S_gated = np.maximum(S - noise_floor * 1.5, 0)
S_gated_db = librosa.amplitude_to_db(S_gated, ref=np.max)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
img0 = librosa.display.specshow(S_db_raw, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=axes[0])
axes[0].set_title('Raw')
fig.colorbar(img0, ax=axes[0], format='%+2.0f dB')
img1 = librosa.display.specshow(S_gated_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=axes[1])
axes[1].set_title('Spectral gating')
fig.colorbar(img1, ax=axes[1], format='%+2.0f dB')
plt.tight_layout()
plt.show()

# %% Denoising method 2: Median filter (removes speckle)
S_median = median_filter(S, size=(3, 5))
S_median_db = librosa.amplitude_to_db(S_median, ref=np.max)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
img0 = librosa.display.specshow(S_db_raw, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=axes[0])
axes[0].set_title('Raw')
fig.colorbar(img0, ax=axes[0], format='%+2.0f dB')
img1 = librosa.display.specshow(S_median_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=axes[1])
axes[1].set_title('Median filter (3x5)')
fig.colorbar(img1, ax=axes[1], format='%+2.0f dB')
plt.tight_layout()
plt.show()

# %% Denoising method 3: Soft mask with nearest-neighbor filter (best for birdsong)
S_filter = librosa.decompose.nn_filter(
    S, aggregate=np.median, metric='cosine',
    width=int(librosa.time_to_frames(0.5, sr=sr, hop_length=hop_length))
)
S_filter = np.minimum(S, S_filter)

mask = librosa.util.softmask(
    S - S_filter,
    10 * S_filter,
    power=2,
    split_zeros=False,
)
S_softmask = mask * S
S_softmask_db = librosa.amplitude_to_db(S_softmask, ref=np.max)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
img0 = librosa.display.specshow(S_db_raw, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=axes[0])
axes[0].set_title('Raw')
fig.colorbar(img0, ax=axes[0], format='%+2.0f dB')
img1 = librosa.display.specshow(S_softmask_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=axes[1])
axes[1].set_title('Soft mask (nn_filter)')
fig.colorbar(img1, ax=axes[1], format='%+2.0f dB')
plt.tight_layout()
plt.show()

# %% Denoising method 4: Combined -- soft mask + light median
S_combined = median_filter(S_softmask, size=(2, 3))
S_combined_db = librosa.amplitude_to_db(S_combined, ref=np.max)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
imgs = []
for ax, data, title in zip(axes,
    [S_db_raw, S_softmask_db, S_combined_db],
    ['Raw', 'Soft mask only', 'Soft mask + median']):
    im = librosa.display.specshow(data, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, format='%+2.0f dB')
plt.tight_layout()
plt.show()

# %% Compare all denoise methods with green colormap (same gamma/percentile)
def to_green_img(S_in, gamma=0.4, pct=99.5):
    db = librosa.amplitude_to_db(S_in, ref=np.max)
    n = (db - db.min()) / (db.max() - db.min() + 1e-8)
    n = np.power(n, gamma)
    p = np.percentile(n, pct)
    n = np.clip(n / (p + 1e-8), 0, 1)
    n = n[::-1]
    h, w = n.shape
    img = np.zeros((h, w, 3))
    img[:, :, 1] = n
    return img

methods = {
    'Raw (no denoise)': S,
    'Spectral gating': S_gated,
    'Median filter': S_median,
    'Soft mask': S_softmask,
    'Soft mask + median': S_combined,
}

fig, axes = plt.subplots(1, len(methods), figsize=(24, 6))
for ax, (title, S_m) in zip(axes, methods.items()):
    ax.imshow(to_green_img(S_m), aspect='auto')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle('All denoise methods (green, gamma=0.4, p=99.5)', fontsize=13)
plt.tight_layout()
plt.show()

# %% Pick your preferred denoised signal -- change this to try different ones
S_clean = S_combined  # <-- swap to S_gated, S_median, S_softmask, or S_combined
S_clean_db = librosa.amplitude_to_db(S_clean, ref=np.max)

# %% Contrast curves comparison
S_norm = (S_clean_db - S_clean_db.min()) / (S_clean_db.max() - S_clean_db.min() + 1e-8)

gamma_values = [0.3, 0.4, 0.5, 0.7]
fig, axes = plt.subplots(1, len(gamma_values), figsize=(20, 5))
for ax, gamma in zip(axes, gamma_values):
    S_g = np.power(S_norm, gamma)
    ax.imshow(S_g[::-1], aspect='auto', cmap='Greens', vmin=0, vmax=1)
    ax.set_title(f'gamma={gamma}')
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle('Gamma comparison (Greens cmap)')
plt.tight_layout()
plt.show()

# %% Percentile stretch comparison
gamma = 0.4  # <-- pick your preferred gamma from above
S_gamma = np.power(S_norm, gamma)

percentiles = [98, 99, 99.5, 99.9]
fig, axes = plt.subplots(1, len(percentiles), figsize=(20, 5))
for ax, pct in zip(axes, percentiles):
    p_hi = np.percentile(S_gamma, pct)
    S_stretched = np.clip(S_gamma / (p_hi + 1e-8), 0, 1)
    ax.imshow(S_stretched[::-1], aspect='auto', cmap='Greens', vmin=0, vmax=1)
    ax.set_title(f'p={pct} (top {100-pct:.1f}% -> white)')
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle(f'Percentile stretch (gamma={gamma})')
plt.tight_layout()
plt.show()

# %% Color mapping: green-only vs green-to-white
gamma = 0.4
p_hi = np.percentile(np.power(S_norm, gamma), 99.5)
S_final = np.clip(np.power(S_norm, gamma) / (p_hi + 1e-8), 0, 1)
S_final = S_final[::-1]

h, w = S_final.shape

img_green = np.zeros((h, w, 3))
img_green[:, :, 1] = S_final

img_warm = np.zeros((h, w, 3))
img_warm[:, :, 1] = S_final
img_warm[:, :, 0] = np.clip((S_final - 0.6) * 2.5, 0, 1) * 0.7
img_warm[:, :, 2] = np.clip((S_final - 0.7) * 3.3, 0, 1) * 0.4

img_cyan = np.zeros((h, w, 3))
img_cyan[:, :, 1] = S_final
img_cyan[:, :, 2] = np.clip((S_final - 0.5) * 2, 0, 1) * 0.6

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, im, title in zip(axes,
    [img_green, img_warm, img_cyan],
    ['Pure green', 'Green + warm white', 'Green + cyan']):
    ax.imshow(im, aspect='auto')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle('Color mapping comparison')
plt.tight_layout()
plt.show()

# %% Final composite -- tweak these parameters, then copy to main.py
DENOISE = 'combined'      # 'none', 'gated', 'median', 'softmask', 'combined'
GAMMA = 0.4
PERCENTILE = 99.5
COLOR = 'green'            # 'green', 'warm', 'cyan'

S_use = {
    'none': S,
    'gated': S_gated,
    'median': S_median,
    'softmask': S_softmask,
    'combined': S_combined,
}[DENOISE]

S_use_db = librosa.amplitude_to_db(S_use, ref=np.max)
S_n = (S_use_db - S_use_db.min()) / (S_use_db.max() - S_use_db.min() + 1e-8)
S_n = np.power(S_n, GAMMA)
p = np.percentile(S_n, PERCENTILE)
S_n = np.clip(S_n / (p + 1e-8), 0, 1)
S_n = S_n[::-1]

h, w = S_n.shape
img = np.zeros((h, w, 3))
img[:, :, 1] = S_n

if COLOR == 'warm':
    img[:, :, 0] = np.clip((S_n - 0.6) * 2.5, 0, 1) * 0.7
    img[:, :, 2] = np.clip((S_n - 0.7) * 3.3, 0, 1) * 0.4
elif COLOR == 'cyan':
    img[:, :, 2] = np.clip((S_n - 0.5) * 2, 0, 1) * 0.6

fig, ax = plt.subplots(figsize=(16, 8))
ax.imshow(img, aspect='auto')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(f'Final: denoise={DENOISE}, gamma={GAMMA}, p={PERCENTILE}, color={COLOR}')
plt.tight_layout()
plt.show()

print(f"\nSettings to copy into main.py:")
print(f"  DENOISE = '{DENOISE}'")
print(f"  GAMMA = {GAMMA}")
print(f"  PERCENTILE = {PERCENTILE}")
print(f"  COLOR = '{COLOR}'")

# %%
