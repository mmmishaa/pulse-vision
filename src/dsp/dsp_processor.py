import numpy as np
import scipy
from collections import deque

class PulseDSP:
    def __init__(self, len_buffer = 300, filter_cutoff_frequencies = [0.75, 2.5]):
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
        raw_data = np.array(self.buffer)[:, 0]
        bpm = compute_bpm(raw_data=raw_data, fps=fps, filter_cutoff_frequencies=self.filter_cutoff_frequencies)

        return bpm

def compute_bpm(raw_data: np.ndarray, fps: float, filter_cutoff_frequencies):
    filtered_data = apply_filter(raw_data=raw_data, fps=fps, filter_cutoff_frequencies=filter_cutoff_frequencies)
    amplitudes, freqs = compute_fft(filtered_data=filtered_data, fps=fps)
    mask = (freqs >= filter_cutoff_frequencies[0]) & (freqs <= filter_cutoff_frequencies[1])
    val_amps, val_freqs = amplitudes[mask], freqs[mask]

    best_idx = np.argmax(val_amps)
    best_freq = val_freqs[best_idx]
    bpm = best_freq * 60

    return bpm

def compute_fft(filtered_data: np.ndarray, fps, n: int = 2048):
    fft_values = np.fft.rfft(a=filtered_data, n=n)
    amplitudes = np.abs(fft_values)
    freqs = np.fft.rfftfreq(n=n, d=1/fps)
    return amplitudes, freqs

def apply_filter(raw_data: np.ndarray, fps: float, filter_cutoff_frequencies) -> np.ndarray:
    b, a = scipy.signal.butter(N=3, Wn=filter_cutoff_frequencies, btype='bandpass', fs=fps)
    filtered_data = scipy.signal.filtfilt(b, a, raw_data)
    return filtered_data

def estimate_bpm_from_wave(wave: np.ndarray, fps: int):
    filter_cutoff_frequencies=[0.75, 2.5]
    bpm = compute_bpm(raw_data=wave, fps=fps, filter_cutoff_frequencies=filter_cutoff_frequencies)
    return bpm

def estimate_bpm_from_diff_wave(diff_wave: np.ndarray, fps: int):
    wave = np.cumsum(diff_wave)
    bpm = estimate_bpm_from_wave(wave=wave, fps=fps)
    return bpm