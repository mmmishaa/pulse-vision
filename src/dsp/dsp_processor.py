import numpy as np
import scipy
from collections import deque

class PulseDSP:
    def __init__(self, len_buffer = 300, filter_cutoff_frequencies = [0.7, 3.0]):
        self.len_buffer = len_buffer
        self.buffer = deque(maxlen=self.len_buffer)
        self.filter_cutoff_frequencies = filter_cutoff_frequencies

    def crop_roi(self, frame: np.ndarray, roi_box: tuple[float, float, float, float]) -> np.ndarray:
        if roi_box is None:
            return None
        x0, y0, x1, y1 = roi_box
        return frame[y0:y1, x0:x1]

    def get_green_mean(self, crop_roi: np.ndarray) -> float:
        if crop_roi is None or crop_roi.size == 0:
            return None
        return float(np.mean(crop_roi[:, :, 1]))

    def apply_filter(self, raw_data: np.ndarray, fps: float) -> np.ndarray:
        b, a = scipy.signal.butter(N=3, Wn=self.filter_cutoff_frequencies, btype='bandpass', fs=fps)
        filtered_data = scipy.signal.filtfilt(b, a, raw_data)

        return filtered_data

    def compute_fft(self, filtered_data: np.ndarray, fps, n: int = 1000):
        fft_values = np.fft.rfft(a=filtered_data, n=n)
        amplitudes = np.abs(fft_values)
        freqs = np.fft.rfftfreq(n=n, d=1/fps)

        return amplitudes, freqs

    def compute_bpm(self, fps: float):
        raw_data = np.array(self.buffer)[:, 0]
        filtered_data = self.apply_filter(raw_data=raw_data, fps=fps)
        amplitudes, freqs = self.compute_fft(filtered_data=filtered_data, fps=fps)
        mask = (freqs >= self.filter_cutoff_frequencies[0]) & (freqs <= self.filter_cutoff_frequencies[1])
        val_amps, val_freqs = amplitudes[mask], freqs[mask]

        best_idx = np.argmax(val_amps)
        best_freq = val_freqs[best_idx]
        bpm = best_freq * 60

        print(f"DEBUG: FPS={fps:.2f}, Max Amp={val_amps[best_idx]:.2f}, Best Freq={best_freq:.2f}")

        return bpm

    def update(self, frame: np.ndarray, curr_ts_ms: float, roi_box: tuple[float, float, float, float]) -> float | None:
        crop_roi = self.crop_roi(frame=frame, roi_box=roi_box)
        gm = self.get_green_mean(crop_roi=crop_roi)
        if gm is None:
            return None
        self.buffer.append((gm, curr_ts_ms))

        if len(self.buffer) < self.len_buffer:
            return None

        duration_seconds = (self.buffer[-1][1] - self.buffer[0][1]) / 1000.0
        fps = len(self.buffer) / duration_seconds
        bpm = self.compute_bpm(fps=fps)

        return bpm