"""
Interactive Birdsong Spectrogram Viewer.

Usage:
    python viewer.py                       # loads first file in ../sound/
    python viewer.py japu.mp3              # loads a specific file

Controls:
    - Click and drag on the spectrogram to select a time region (cyan box)
    - Play: plays the selected region on loop with a moving bar
    - Stop: stops playback
    - Reset: clears selection, shows full spectrogram
    - Threshold slider: drag to cut noise floor
    - Scroll wheel: zoom in/out on time axis (centered on cursor)
    - Right-click drag: pan the view
"""

import sys
import threading
import time
from pathlib import Path

import librosa
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle
import numpy as np
import sounddevice as sd

SOUND_DIR = Path(__file__).resolve().parent.parent / "sound"


class SpectrogramViewer:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.playing = False
        self.looping = True
        self.play_thread = None
        self.bar_line = None

        self.sel_t0 = None
        self.sel_t1 = None
        self.sel_rect = None
        self._drag_start = None
        self._pan_start = None

        self._load_audio()
        self._build_ui()
        self._connect_events()

    def _load_audio(self):
        print(f"Loading {self.audio_path} ...")
        self.y, self.sr = librosa.load(str(self.audio_path), sr=None, mono=True)

        n_fft = 2048
        hop_length = 512
        self.S = np.abs(librosa.stft(self.y, n_fft=n_fft, hop_length=hop_length))
        self.S_db = librosa.amplitude_to_db(self.S, ref=np.max)
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        self.times = librosa.frames_to_time(
            np.arange(self.S.shape[1]), sr=self.sr, hop_length=hop_length
        )
        self.vmin = float(self.S_db.min())
        self.vmax = float(self.S_db.max())
        self.duration = len(self.y) / self.sr
        self.full_tlim = (self.times[0], self.times[-1])
        self.full_flim = (self.freqs[0], self.freqs[-1])
        print(f"  Duration: {self.duration:.1f}s | sr: {self.sr} | shape: {self.S.shape}")

    def _build_ui(self):
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.patch.set_facecolor('#1a1a1a')

        self.ax = self.fig.add_axes([0.06, 0.18, 0.88, 0.75])
        self.ax.set_facecolor('#000')

        self.img = self.ax.imshow(
            self.S_db, aspect='auto', origin='lower',
            extent=[self.times[0], self.times[-1], self.freqs[0], self.freqs[-1]],
            cmap='magma', vmin=self.vmin, vmax=self.vmax,
            interpolation='nearest'
        )
        self.ax.set_xlabel('Time (s)', color='#ccc')
        self.ax.set_ylabel('Frequency (Hz)', color='#ccc')
        self.ax.set_title(
            f'{Path(self.audio_path).name}  —  drag to select, scroll to zoom, right-drag to pan',
            color='#aaa', fontsize=11, pad=8
        )
        self.ax.tick_params(colors='#999')
        for spine in self.ax.spines.values():
            spine.set_color('#444')

        self.bar_line = self.ax.axvline(x=-1, color='white', linewidth=1.5, alpha=0.9)
        self.bar_line.set_visible(False)

        # Threshold slider
        ax_thresh = self.fig.add_axes([0.06, 0.05, 0.45, 0.03])
        ax_thresh.set_facecolor('#2a2a2a')
        self.slider = Slider(
            ax_thresh, 'Threshold (dB)', self.vmin, self.vmax,
            valinit=self.vmin, valstep=0.5, color='#b5446e'
        )
        self.slider.label.set_color('#aaa')
        self.slider.valtext.set_color('#aaa')
        self.slider.on_changed(self._on_thresh_change)

        # Buttons
        ax_play = self.fig.add_axes([0.58, 0.04, 0.1, 0.05])
        self.btn_play = Button(ax_play, '▶ Loop', color='#333', hovercolor='#555')
        self.btn_play.label.set_color('#0f0')
        self.btn_play.on_clicked(self._on_play)

        ax_once = self.fig.add_axes([0.69, 0.04, 0.1, 0.05])
        self.btn_once = Button(ax_once, '▶ Once', color='#333', hovercolor='#555')
        self.btn_once.label.set_color('#8f8')
        self.btn_once.on_clicked(self._on_play_once)

        ax_stop = self.fig.add_axes([0.80, 0.04, 0.08, 0.05])
        self.btn_stop = Button(ax_stop, '■ Stop', color='#333', hovercolor='#555')
        self.btn_stop.label.set_color('#f44')
        self.btn_stop.on_clicked(self._on_stop)

        ax_reset = self.fig.add_axes([0.89, 0.04, 0.08, 0.05])
        self.btn_reset = Button(ax_reset, 'Reset', color='#333', hovercolor='#555')
        self.btn_reset.label.set_color('#aaa')
        self.btn_reset.on_clicked(self._on_reset)

    def _connect_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('close_event', self._on_close)

    # --- Mouse interaction ---

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self._drag_start = event.xdata
            self._clear_selection()
        elif event.button == 3:
            self._pan_start = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if event.inaxes != self.ax:
            return
        if self._drag_start is not None and event.button == 1:
            t0 = min(self._drag_start, event.xdata)
            t1 = max(self._drag_start, event.xdata)
            f0, f1 = self.ax.get_ylim()
            self._draw_selection(t0, t1, f0, f1)
        elif self._pan_start is not None and event.button == 3:
            dx = self._pan_start[0] - event.xdata
            dy = self._pan_start[1] - event.ydata
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            self.ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
            self.ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
            self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if event.button == 1 and self._drag_start is not None:
            if event.inaxes == self.ax and event.xdata is not None:
                t0 = min(self._drag_start, event.xdata)
                t1 = max(self._drag_start, event.xdata)
                if t1 - t0 > 0.05:
                    self.sel_t0 = max(0, t0)
                    self.sel_t1 = min(self.duration, t1)
                    print(f"Selected: [{self.sel_t0:.2f}, {self.sel_t1:.2f}]s  "
                          f"({self.sel_t1 - self.sel_t0:.2f}s)")
                else:
                    self._clear_selection()
            self._drag_start = None
        elif event.button == 3:
            self._pan_start = None

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        factor = 0.8 if event.button == 'up' else 1.25
        xlim = self.ax.get_xlim()
        xc = event.xdata
        new_half = (xlim[1] - xlim[0]) * factor / 2
        self.ax.set_xlim(xc - new_half, xc + new_half)

        ylim = self.ax.get_ylim()
        yc = event.ydata
        new_half_y = (ylim[1] - ylim[0]) * factor / 2
        self.ax.set_ylim(yc - new_half_y, yc + new_half_y)

        self.fig.canvas.draw_idle()

    def _draw_selection(self, t0, t1, f0, f1):
        if self.sel_rect:
            self.sel_rect.remove()
        self.sel_rect = Rectangle(
            (t0, f0), t1 - t0, f1 - f0,
            linewidth=1.5, edgecolor='cyan', facecolor='cyan', alpha=0.12
        )
        self.ax.add_patch(self.sel_rect)
        self.fig.canvas.draw_idle()

    def _clear_selection(self):
        if self.sel_rect:
            self.sel_rect.remove()
            self.sel_rect = None
        self.sel_t0 = None
        self.sel_t1 = None
        self.fig.canvas.draw_idle()

    # --- Threshold ---

    def _on_thresh_change(self, val):
        self.img.set_clim(vmin=val)
        self.fig.canvas.draw_idle()

    # --- Playback ---

    def _get_play_range(self):
        if self.sel_t0 is not None and self.sel_t1 is not None:
            return self.sel_t0, self.sel_t1
        t0, t1 = self.ax.get_xlim()
        return max(0, t0), min(self.duration, t1)

    def _on_play(self, event):
        self._start_playback(loop=True)

    def _on_play_once(self, event):
        self._start_playback(loop=False)

    def _start_playback(self, loop):
        if self.playing:
            self._on_stop(None)
            time.sleep(0.05)

        t0, t1 = self._get_play_range()
        if t1 <= t0:
            return

        self.playing = True
        self.looping = loop

        s0 = int(t0 * self.sr)
        s1 = int(t1 * self.sr)
        self._play_chunk = self.y[s0:s1].copy()
        self._play_pos = 0
        self._play_t0 = t0
        self._play_t1 = t1
        self._play_chunk_len = len(self._play_chunk)
        self._play_start_wall = None

        self._stream = sd.OutputStream(
            samplerate=self.sr, channels=1, dtype='float32',
            callback=self._audio_callback, blocksize=1024
        )
        self._play_start_wall = time.monotonic()
        self._stream.start()

        self.play_thread = threading.Thread(
            target=self._animate_loop, daemon=True
        )
        self.play_thread.start()

    def _audio_callback(self, outdata, frames, time_info, status):
        chunk = self._play_chunk
        n = self._play_chunk_len
        pos = self._play_pos
        end = pos + frames

        if end <= n:
            outdata[:, 0] = chunk[pos:end]
            self._play_pos = end
        else:
            remaining = n - pos
            if remaining > 0:
                outdata[:remaining, 0] = chunk[pos:n]
            if self.looping:
                self._play_pos = (end - n) % n
                self._play_start_wall = time.monotonic() - self._play_pos / self.sr
                outdata[remaining:, 0] = chunk[:frames - remaining]
            else:
                outdata[remaining:, 0] = 0
                self.playing = False

    def _animate_loop(self):
        t0 = self._play_t0
        chunk_dur = self._play_t1 - t0
        self.bar_line.set_visible(True)

        while self.playing:
            elapsed = time.monotonic() - self._play_start_wall
            if self.looping:
                elapsed = elapsed % chunk_dur
            else:
                elapsed = min(elapsed, chunk_dur)
            t = t0 + elapsed
            self.bar_line.set_xdata([t, t])
            try:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            except Exception:
                break
            time.sleep(0.016)

        self.bar_line.set_visible(False)
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _on_stop(self, event):
        self.playing = False
        if hasattr(self, '_stream') and self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.bar_line.set_visible(False)
        self.fig.canvas.draw_idle()

    def _on_reset(self, event):
        self._on_stop(None)
        self._clear_selection()
        self.ax.set_xlim(self.full_tlim)
        self.ax.set_ylim(self.full_flim)
        self.slider.set_val(self.vmin)
        self.fig.canvas.draw_idle()

    def _on_close(self, event):
        self.playing = False
        if hasattr(self, '_stream') and self._stream:
            self._stream.stop()
            self._stream.close()

    def show(self):
        plt.show()


def main():
    if len(sys.argv) > 1:
        name = sys.argv[1]
        path = SOUND_DIR / name
    else:
        exts = {'.mp3', '.wav', '.flac', '.ogg'}
        files = sorted(f for f in SOUND_DIR.iterdir() if f.suffix.lower() in exts)
        if not files:
            print(f"No audio files in {SOUND_DIR}")
            sys.exit(1)
        path = files[0]

    if not path.is_file():
        print(f"File not found: {path}")
        sys.exit(1)

    viewer = SpectrogramViewer(path)
    viewer.show()


if __name__ == '__main__':
    main()
