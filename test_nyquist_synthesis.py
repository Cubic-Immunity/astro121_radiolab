#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
matplotlib.use('Agg')  # Disable GUI for headless testing
import pytest 
from functions import plot_nyquist_synthesis


# In[ ]:


# Run tests with 'pytest -v test_nyquist_synthesis.ipynb' in terminal or in notebook with ! in front
def make_sine(fs, f_sig, N=2048, amp=1.0):
    """Generate a test sine wave."""
    t = np.arange(N) / fs
    return amp * np.sin(2 * np.pi * f_sig * t)

def test_valid_input_returns_results():
    fs = 1e6
    f_sig = 500e3
    data_dict = {
        0: {
            'sample_rate': fs,
            'filename': 'test_signal.npz',
            'data': np.array([make_sine(fs, f_sig)])
        }
    }
    results = plot_nyquist_synthesis(data_dict, true_freq_khz=f_sig/1e3, show_individual=False)
    assert isinstance(results, list)
    assert len(results) == 1
    assert np.isclose(results[0]['sample_rate'], fs)


# In[ ]:




