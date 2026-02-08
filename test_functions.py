#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib
matplotlib.use('Agg')
import pytest
from scipy.signal import correlate

from functions import (
    load_lab_data,
    plot_nyquist_synthesis,
    analyze_voltage_spectrum_symmetry,
    analyze_acf_vs_ifft,
    plot_spectral_leakage_multiview,
    nyquist_windows_plot
)


# In[4]:


# Run tests with 'pytest -v test_nyquist_synthesis.ipynb' in terminal or in notebook with ! at the start
# Helper data
def make_sine(fs=1e6, f_sig=100e3, N=2048, amp=1.0):
    t = np.arange(N) / fs
    return amp * np.sin(2 * np.pi * f_sig * t)

def make_fake_data_dict(fs=1e6, f_sig=100e3):
    data = np.array([make_sine(fs, f_sig)])
    return {0: {"filename": "fake_signal.npz", "data": data, "sample_rate": fs, "meta": {}}}

# Functions
def test_load_lab_data_with_fake_npz(tmp_path):
    """Test loading .npz data with 'data' key."""
    fname = tmp_path / "fake_1MHz_data.npz"
    np.savez(fname, data=np.random.randn(3, 5, 2048), sample_rate_mhz=1.0)
    data_dict = load_lab_data([str(fname)])
    assert isinstance(data_dict, dict)
    assert len(data_dict) == 1
    key = list(data_dict.keys())[0]
    entry = data_dict[key]
    assert "data" in entry
    assert entry["sample_rate"] == 1e6
    assert entry["data"].ndim == 2  # (blocks, samples)

def test_plot_nyquist_synthesis_runs_and_returns_results():
    """Ensure Nyquist synthesis runs and produces expected dict keys."""
    fs = 1e6
    f_sig = 300e3
    data_dict = make_fake_data_dict(fs, f_sig)
    results = plot_nyquist_synthesis(data_dict, true_freq_khz=f_sig/1e3)
    assert isinstance(results, list)
    r = results[0]
    for field in ["filename", "sample_rate", "nyquist_zone", "measured_freq"]:
        assert field in r

def test_analyze_voltage_spectrum_symmetry_executes(monkeypatch):
    """Ensure symmetry plot runs without error."""
    data_dict = make_fake_data_dict()
    entry = data_dict[0]
    # Patch plt.show() to suppress display
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    analyze_voltage_spectrum_symmetry(entry, block_idx=0)
    # Basic check
    fs = entry["sample_rate"]
    assert np.isclose(fs, 1e6)

def test_analyze_acf_vs_ifft_correctness(monkeypatch):
    """Verify ACF vs IFFT correlation consistency."""
    fs = 1e6
    data_dict = make_fake_data_dict(fs)
    entry = data_dict[0]
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    result = analyze_acf_vs_ifft(entry, zoom=0.01)
    # Ensure outputs exist and shapes match
    assert all(k in result for k in ["acf_np", "acf_sp", "acf_freq", "lags"])
    assert len(result["acf_np"]) == len(result["acf_sp"]) == len(result["acf_freq"])
    # Check correlation theorem roughly holds
    diff = np.max(np.abs(result["acf_np"] - result["acf_freq"]))
    assert diff < 0.2  # loose tolerance since windowing differs

def test_plot_spectral_leakage_multiview_executes(monkeypatch):
    """Verify spectral leakage visualization runs."""
    data_dict = make_fake_data_dict(fs=1e6, f_sig=100e3)
    entry = data_dict[0]
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    plot_spectral_leakage_multiview(entry, freq_dense_points=2001, statistics=True)
    assert entry["data"].shape[1] == 2048

def test_nyquist_windows_plot_executes(monkeypatch):
    """Ensure Nyquist window plotting executes and returns correct fields."""
    data_dict = make_fake_data_dict(fs=1e6, f_sig=100e3)
    entry = data_dict[0]
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    result = nyquist_windows_plot(entry, W=2, Nfreq=1024)
    assert all(k in result for k in ["freqs", "power", "window_edges"])
    assert isinstance(result["freqs"], np.ndarray)
    assert isinstance(result["power"], np.ndarray)
    assert len(result["window_edges"]) == 5


# In[ ]:




