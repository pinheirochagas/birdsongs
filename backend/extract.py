# %% [markdown]
# # Birdsong Pattern Extractor
# Interactive tool to zoom, threshold, and extract distinct birdsong patterns.

# %% Load audio
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RectangleSelector
from matplotlib.patches import Rectangle
from scipy.ndimage import median_filter
from pathlib import Path

SOUND_DIR = Path(__file__).resolve().parent.parent / "sound"
AUDIO_FILE = "japu.mp3"
AUDIO_PATH = str(SOUND_DIR / AUDIO_FILE)

y, sr = librosa.load(AUDIO_PATH, sr=None, mono=True)
n_fft = 2048
hop_length = 512

S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)

S_db = librosa.amplitude_to_db(S, ref=np.max)

print(f"Loaded: {AUDIO_FILE}")
print(f"Duration: {len(y)/sr:.1f}s | sr: {sr} | Spectrogram: {S.shape}")

# %% Interactive spectrogram with threshold slider
# Use matplotlib's built-in zoom/pan toolbar + a threshold slider.
# Zoom in with the magnifying glass, then adjust the threshold to isolate patterns.

fig, (ax_spec, ax_thresh) = plt.subplots(
    2, 1, figsize=(16, 8),
    gridspec_kw={'height_ratios': [6, 1], 'hspace': 0.25}
)

vmin, vmax = S_db.min(), S_db.max()
img = ax_spec.imshow(
    S_db, aspect='auto', origin='lower',
    extent=[times[0], times[-1], freqs[0], freqs[-1]],
    cmap='magma', vmin=vmin, vmax=vmax,
    interpolation='nearest'
)
ax_spec.set_xlabel('Time (s)')
ax_spec.set_ylabel('Frequency (Hz)')
ax_spec.set_title(f'{AUDIO_FILE} — use toolbar to zoom, slider to threshold')
fig.colorbar(img, ax=ax_spec, format='%+.0f dB', pad=0.01)

slider_thresh = Slider(
    ax_thresh, 'Floor (dB)', vmin, vmax,
    valinit=vmin, valstep=0.5, color='#b5446e'
)

def update_thresh(val):
    img.set_clim(vmin=val)
    fig.canvas.draw_idle()

slider_thresh.on_changed(update_thresh)
plt.show()

# %% Zoom into a region — set bounds and re-run this cell
# Change these values to zoom. Set to None for full range.

T_MIN, T_MAX = 3, 8       # time in seconds
F_MIN, F_MAX = 0, 12000   # frequency in Hz
THRESH_DB = -40            # dB floor

t_mask = (times >= T_MIN) & (times <= T_MAX)
f_mask = (freqs >= F_MIN) & (freqs <= F_MAX)
patch = S_db[np.ix_(f_mask, t_mask)]

fig, ax = plt.subplots(figsize=(14, 6))
ax.imshow(
    patch, aspect='auto', origin='lower',
    extent=[T_MIN, T_MAX, F_MIN, F_MAX],
    cmap='magma', vmin=THRESH_DB, vmax=vmax,
    interpolation='nearest'
)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
ax.set_title(f'Zoom: t=[{T_MIN}, {T_MAX}]s  f=[{F_MIN}, {F_MAX}]Hz  floor={THRESH_DB} dB')
fig.colorbar(ax.images[0], ax=ax, format='%+.0f dB', pad=0.01)
plt.tight_layout()
plt.show()

# %% Binary mask view — see what survives the threshold

mask = (S_db >= THRESH_DB).astype(float)
mask_clean = median_filter(mask, size=(5, 5))

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].imshow(
    S_db, aspect='auto', origin='lower',
    extent=[times[0], times[-1], freqs[0], freqs[-1]],
    cmap='magma', vmin=THRESH_DB, vmax=vmax
)
axes[0].set_title(f'Thresholded spectrogram (floor={THRESH_DB} dB)')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Frequency (Hz)')

axes[1].imshow(
    mask_clean, aspect='auto', origin='lower',
    extent=[times[0], times[-1], freqs[0], freqs[-1]],
    cmap='gray'
)
axes[1].set_title('Binary mask (cleaned)')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Frequency (Hz)')

plt.tight_layout()
plt.show()

# %% Region selector — drag rectangles to mark birdsong regions
# Click and drag to select a time-frequency region.
# Each region is stored in `regions` list.

regions = []

fig, ax = plt.subplots(figsize=(16, 6))
ax.imshow(
    S_db, aspect='auto', origin='lower',
    extent=[times[0], times[-1], freqs[0], freqs[-1]],
    cmap='magma', vmin=THRESH_DB, vmax=vmax
)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
ax.set_title('Drag to select birdsong regions (each box is saved)')

def on_select(eclick, erelease):
    t0, t1 = sorted([eclick.xdata, erelease.xdata])
    f0, f1 = sorted([eclick.ydata, erelease.ydata])
    regions.append({'t0': t0, 't1': t1, 'f0': f0, 'f1': f1})
    rect = Rectangle(
        (t0, f0), t1 - t0, f1 - f0,
        linewidth=1.5, edgecolor='cyan', facecolor='cyan', alpha=0.15
    )
    ax.add_patch(rect)
    ax.text(t0, f1, f'#{len(regions)}', color='cyan', fontsize=9,
            va='bottom', fontweight='bold')
    fig.canvas.draw_idle()
    print(f"Region #{len(regions)}: t=[{t0:.2f}, {t1:.2f}]s  f=[{f0:.0f}, {f1:.0f}]Hz")

selector = RectangleSelector(
    ax, on_select, useblit=True,
    button=[1], interactive=False,
    props=dict(edgecolor='cyan', alpha=0.3, fill=True)
)
plt.show()

# %% View extracted regions
# Shows each selected region as a zoomed spectrogram.

if not regions:
    print("No regions selected — run the cell above and drag some boxes first.")
else:
    n = len(regions)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for i, reg in enumerate(regions):
        ax = axes[i // cols][i % cols]
        t_mask = (times >= reg['t0']) & (times <= reg['t1'])
        f_mask = (freqs >= reg['f0']) & (freqs <= reg['f1'])
        patch = S_db[np.ix_(f_mask, t_mask)]

        ax.imshow(
            patch, aspect='auto', origin='lower',
            extent=[reg['t0'], reg['t1'], reg['f0'], reg['f1']],
            cmap='magma', vmin=THRESH_DB, vmax=vmax
        )
        ax.set_title(f"Region #{i+1}", fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Freq (Hz)')

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis('off')

    plt.tight_layout()
    plt.show()

# %% Extract isolated patterns using connected components
# Finds distinct blobs in the thresholded spectrogram.

from scipy.ndimage import label, find_objects

mask = (S_db >= THRESH_DB).astype(np.int32)
mask = median_filter(mask, size=(5, 5))

labeled, n_features = label(mask)
print(f"Found {n_features} connected components at threshold {THRESH_DB} dB")

slices = find_objects(labeled)
# Filter out tiny fragments (min area in pixels)
MIN_AREA = 200
patterns = []
for i, sl in enumerate(slices):
    if sl is None:
        continue
    region_mask = labeled[sl] == (i + 1)
    area = region_mask.sum()
    if area >= MIN_AREA:
        patterns.append({
            'id': i + 1,
            'freq_slice': sl[0],
            'time_slice': sl[1],
            'area': area,
            't0': times[sl[1].start] if sl[1].start < len(times) else times[-1],
            't1': times[min(sl[1].stop - 1, len(times) - 1)],
            'f0': freqs[sl[0].start],
            'f1': freqs[min(sl[0].stop - 1, len(freqs) - 1)],
        })

print(f"Kept {len(patterns)} patterns with area >= {MIN_AREA} pixels")
for p in patterns:
    print(f"  Pattern {p['id']:3d}: t=[{p['t0']:.2f}, {p['t1']:.2f}]s  "
          f"f=[{p['f0']:.0f}, {p['f1']:.0f}]Hz  area={p['area']}")

# %% Plot the largest patterns
# Shows the top N patterns sorted by area.

TOP_N = 12

top = sorted(patterns, key=lambda p: p['area'], reverse=True)[:TOP_N]
cols = min(len(top), 4)
rows = (len(top) + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)

for i, p in enumerate(top):
    ax = axes[i // cols][i % cols]
    patch = S_db[p['freq_slice'], p['time_slice']]
    component_mask = labeled[p['freq_slice'], p['time_slice']] == p['id']
    masked_patch = np.where(component_mask, patch, np.nan)

    ax.imshow(
        masked_patch, aspect='auto', origin='lower',
        extent=[p['t0'], p['t1'], p['f0'], p['f1']],
        cmap='magma', vmin=THRESH_DB, vmax=vmax
    )
    ax.set_title(f"#{p['id']} (area={p['area']})", fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=8)
    ax.set_ylabel('Freq (Hz)', fontsize=8)

for j in range(len(top), rows * cols):
    axes[j // cols][j % cols].axis('off')

plt.suptitle(f'Top {len(top)} patterns (threshold={THRESH_DB} dB, min_area={MIN_AREA})', fontsize=12)
plt.tight_layout()
plt.show()

# %% Full spectrogram with pattern bounding boxes overlaid

fig, ax = plt.subplots(figsize=(16, 6))
ax.imshow(
    S_db, aspect='auto', origin='lower',
    extent=[times[0], times[-1], freqs[0], freqs[-1]],
    cmap='magma', vmin=THRESH_DB, vmax=vmax
)

for p in top:
    rect = Rectangle(
        (p['t0'], p['f0']),
        p['t1'] - p['t0'], p['f1'] - p['f0'],
        linewidth=1, edgecolor='cyan', facecolor='none'
    )
    ax.add_patch(rect)
    ax.text(p['t0'], p['f1'], f"{p['id']}", color='cyan', fontsize=7, va='bottom')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
ax.set_title(f'Top {len(top)} patterns highlighted')
plt.tight_layout()
plt.show()

# %%
