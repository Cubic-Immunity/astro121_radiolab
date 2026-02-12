#!/usr/bin/env python
# coding: utf-8

# In[1]:


def load_lab_data(file_list):
    """
    Load SDR .npz data files for the lab.
    Handles data shapes such as:
      (3, 10, 2048) -> (runs, blocks, samples)
      (2, 2048)     -> (blocks, samples)
      (2048,)       -> (samples)
    Drops first stale block, keeps first run if multiple runs exist.
    """
    data_dict = {}

    for i, fname in enumerate(file_list):
        arr = np.load(fname, allow_pickle=True)
        keys = list(arr.keys())

        # Detect correct data key
        if 'data' in arr:
            key = 'data'
        elif 'data_direct' in arr:
            key = 'data_direct'
        elif 'data_complex' in arr:
            key = 'data_complex'
        else:
            print(f"{fname}: No usable data key found, skipping.")
            continue

        data = arr[key]

        # Handle shape cases
        if data.ndim == 3:
            # (runs, blocks, samples)
            n_runs, n_blocks, n_samp = data.shape
            print(f"{fname}: shape={data.shape} → keeping run 0, dropping stale block 0")
            data = data[0, 1:, :]   # keep first run, drop first block
        elif data.ndim == 2:
            # (blocks, samples)
            print(f"{fname}: shape={data.shape} → dropping stale block 0")
            data = data[1:, :]
        elif data.ndim == 1:
            # (samples,)
            print(f"{fname}: shape={data.shape} → single block assumed")
            data = data[None, :]
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        # Infer sample rate if present
        if 'sample_rate_mhz' in arr:
            fs = float(arr['sample_rate_mhz']) * 1e6
        else:
            import re
            sr_match = re.search(r'(\d+(?:\.\d+)?)MHz', fname)
            fs = float(sr_match.group(1))*1e6 if sr_match else None

        # Store metadata
        meta = {k: arr[k] for k in arr.keys() if k not in ['data', 'data_direct', 'data_complex']}
        data_dict[i] = {
            'filename': fname,
            'data': data.astype(float),
            'sample_rate': fs,
            'meta': meta
        }

        print(f"Loaded {fname}: key='{key}', runs→{data.shape[0]}, samples/block={data.shape[1]}, fs={fs/1e6 if fs else 'unknown'} MHz")

    return data_dict


# In[2]:


def plot_nyquist_synthesis(data_dict, true_freq_khz=None, npoints=200, 
                          indices_to_plot=None, show_individual=False):
    """
    Plots voltage and power spectra. 
    Synthesize multiple data runs into a single plot to demonstrate Nyquist criterion.
    """

    if indices_to_plot is None:
        indices_to_plot = list(data_dict.keys())

    # Filter valid entries
    valid_entries = []
    for i in indices_to_plot:
        if i in data_dict and data_dict[i]['sample_rate'] is not None:
            valid_entries.append(i)

    if len(valid_entries) == 0:
        print("No valid entries with sample rates found!")
        return

    # Create synthesis plot
    fig, axes = plt.subplots(len(valid_entries), 2, 
                            figsize=(14, 4*len(valid_entries)))

    if len(valid_entries) == 1:
        axes = axes.reshape(1, -1)

    results = []

    for plot_idx, entry_idx in enumerate(valid_entries):
        entry = data_dict[entry_idx]
        fs = entry['sample_rate']
        data_blocks = entry['data']
        filename = entry['filename']

        # Use first valid block
        block = data_blocks[0]
        N = len(block)

        # Time domain
        t = np.arange(N) / fs

        # Frequency domain
        spec = np.fft.fftshift(np.fft.fft(block))
        freq = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
        power = np.abs(spec)**2

        # Find peak frequency (estimate true frequency if not provided)
        peak_idx = np.argmax(power)
        estimated_freq_khz = freq[peak_idx] / 1e3

        # Use true frequency if provided, otherwise use estimated
        display_freq_khz = true_freq_khz if true_freq_khz is not None else estimated_freq_khz

        # Calculate Nyquist zone
        if true_freq_khz is not None:
            nyquist_freq = fs / 2
            zone = int(abs(true_freq_khz * 1e3) / nyquist_freq)

            # Calculate expected aliased frequency
            f_in = true_freq_khz * 1e3
            if zone % 2 == 0:
                f_aliased = f_in - zone * nyquist_freq
            else:
                f_aliased = (zone + 1) * nyquist_freq - f_in

            # Handle sign
            if f_in < 0:
                f_aliased = -f_aliased
        else:
            zone = None
            f_aliased = estimated_freq_khz * 1e3

        # Time domain plot
        axes[plot_idx, 0].plot(t[:npoints]*1e3, block[:npoints], linewidth=1.5)
        axes[plot_idx, 0].set_xlabel('Time (ms)', fontsize=12)
        axes[plot_idx, 0].set_ylabel('Voltage (arb.)', fontsize=12)
        axes[plot_idx, 0].set_title(f'{filename}\nTime Domain', fontsize=11)
        axes[plot_idx, 0].grid(alpha=0.3)

        # Frequency domain plot
        axes[plot_idx, 1].plot(freq/1e3, power, linewidth=1.5)

        # Mark Nyquist limits
        axes[plot_idx, 1].axvline(fs/2/1e3, color='red', ls=':', 
                                 label=f'Nyquist: ±{fs/2/1e3:.0f} kHz', alpha=0.7)
        axes[plot_idx, 1].axvline(-fs/2/1e3, color='red', ls=':', alpha=0.7)

        # Mark input frequency
        if true_freq_khz is not None:
            axes[plot_idx, 1].axvline(true_freq_khz, color='green', ls='--', 
                                     label=f'Input: {true_freq_khz:.0f} kHz', 
                                     linewidth=2, alpha=0.7)

        # Mark measured peak
        axes[plot_idx, 1].axvline(estimated_freq_khz, color='orange', ls=':', 
                                 label=f'Measured: {estimated_freq_khz:.1f} kHz', 
                                 linewidth=2)

        axes[plot_idx, 1].set_xlabel('Frequency (kHz)', fontsize=14)
        axes[plot_idx, 1].set_ylabel('Power (arb.)', fontsize=14)

        title_str = f'{filename}\nfs={fs/1e6:.2f} MHz'
        if zone is not None:
            title_str += f', Nyquist Zone {zone}'
        axes[plot_idx, 1].set_title(title_str, fontsize=16)
        axes[plot_idx, 1].tick_params(labelsize=11)
        axes[plot_idx, 1].legend(fontsize=11)
        axes[plot_idx, 1].grid(alpha=0.3)

        # Store results
        results.append({
            'filename': filename,
            'sample_rate': fs,
            'nyquist_zone': zone,
            'input_freq': true_freq_khz * 1e3 if true_freq_khz else None,
            'measured_freq': estimated_freq_khz * 1e3,
            'expected_aliased': f_aliased,
            'error': abs(estimated_freq_khz * 1e3 - f_aliased) if true_freq_khz else None
        })

    plt.tight_layout()
    plt.show()
    return results


# In[3]:


def analyze_voltage_spectrum_symmetry(data_entry, block_idx=0, zoom_to_peaks=True):
    """
    Analyze hermitian symmetry in voltage spectra for real-valued signals.
    Shows real/imaginary parts.
    """
    fs = data_entry['sample_rate']
    block = data_entry['data'][block_idx]
    N = len(block)

    # Check if data is complex
    is_complex = np.iscomplexobj(block)

    spec = np.fft.fftshift(np.fft.fft(block))
    freq = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))

    # Find peaks to determine zoom range
    power = np.abs(spec)**2
    peak_idx = np.argmax(power)
    peak_freq = freq[peak_idx]

    # Find significant power threshold
    threshold = 0.01 * np.max(power)
    significant_indices = np.where(power > threshold)[0]

    if len(significant_indices) > 0 and zoom_to_peaks:
        # Get frequency range of significant power
        freq_min = freq[significant_indices[0]]
        freq_max = freq[significant_indices[-1]]
        # Add some padding (20% on each side)
        freq_range = freq_max - freq_min
        padding = 0.1 * freq_range if freq_range > 0 else 50e3  # 50 kHz default padding
        xlim = (freq_min - padding, freq_max + padding)
    else:
        xlim = None

    plt.figure(figsize=(12, 5))

    # Real and Imaginary parts together
    plt.plot(freq/1e3, spec.real, linewidth=1.5, label='Real', alpha=0.8)
    plt.plot(freq/1e3, spec.imag, linewidth=1.5, label='Imaginary', 
                alpha=0.8, linestyle='--', color='orange')

    # Mark the peak
    plt.axvline(peak_freq/1e3, color='red', linestyle=':', alpha=0.5, 
               label=f'Peak: {peak_freq/1e3:.1f} kHz')

    if xlim:
        plt.xlim(xlim[0]/1e3, xlim[1]/1e3)

    plt.xlabel('Frequency (kHz)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    plt.title(f'Real and Imaginary Parts of Voltage Spectrum\n{data_entry["filename"]}', 
             fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.show()


# In[4]:


def analyze_acf_vs_ifft(data_entry, zoom=0.05):
    """
    Compute and compare autocorrelation via:
      - time-domain correlation (numpy, scipy)
      - frequency-domain IFFT(power)
    """
    fs = data_entry['sample_rate']
    data = data_entry['data'][0]      # first valid block
    N = len(data)
    M = 2 * N - 1                    # target padded length for linear ACF

    # Pad data before correlation 
    data_padded = np.zeros(M, dtype=float)
    data_padded[:N] = data

    # Compute ACFs in time domain 
    acf_np = np.correlate(data_padded, data_padded, mode='full').astype(float)
    acf_sp = correlate(data_padded, data_padded, mode='full').astype(float)

    # Normalize both 
    acf_np /= np.max(np.abs(acf_np))
    acf_sp /= np.max(np.abs(acf_sp))

    # Compute ACF via IFFT(Power) 
    data_padded_fft = np.zeros(2*M - 1, dtype=float)
    data_padded_fft[:N] = data
    spec_padded = np.fft.fft(data_padded_fft)
    power_padded = np.abs(spec_padded)**2
    acf_freq = np.fft.fftshift(np.fft.ifft(power_padded).real)
    acf_freq /= np.max(np.abs(acf_freq))

    # Lags axis 
    lags = np.arange(-(M - 1), M) / fs

    # Plots
    plt.figure(figsize=(14, 5))
    plt.plot(lags * 1e3, acf_np, label='ACF (numpy.correlate)', linewidth=2, alpha=0.7)
    plt.plot(lags * 1e3, acf_sp, '--', label='ACF (scipy.signal.correlate)', linewidth=2, alpha=0.7)
    plt.plot(lags * 1e3, acf_freq, ':', label='IFFT(Power)(zero-padded)', linewidth=2, alpha=0.7, color='red')
    plt.xlabel('Lag (ms)', fontsize=14)
    plt.ylabel('Normalized ACF', fontsize=14)
    plt.xlim((-2.1, 2.1))
    plt.title('ACF vs Inverse Fourier Transform of Power Spectrum', fontsize=18)
    plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    # Zoomed view around zero lag
    plt.figure(figsize=(14, 5))
    plt.plot(lags * 1e3, acf_np, label='ACF (numpy.correlate)', linewidth=4, alpha=0.7)
    plt.plot(lags * 1e3, acf_sp, '--', label='ACF (scipy.signal.correlate)', linewidth=4, alpha=0.7)
    plt.plot(lags * 1e3, acf_freq, ':', label='IFFT(Power)(zero-padded)', linewidth=4, alpha=0.7, color='red')
    plt.xlim(-zoom, zoom)
    plt.xlabel('Lag (ms)', fontsize=16)
    plt.ylabel('Normalized ACF', fontsize=16)
    plt.title('ACF vs Inverse Fourier Transform (Zoomed)', fontsize=18)
    plt.legend(fontsize=12); plt.grid(); plt.tight_layout(); plt.show()

    # Diagnostic checks 
    print(f"Length of data: {N}")
    print(f"Length of padded data (for correlation): {M}")
    print(f"Length of padded data (for FFT): {2*M - 1}")
    print(f"Length of correlation output: {len(acf_np)}")
    print(f"Length of IFFT output: {len(acf_freq)}")
    print(f"Length of lags: {len(lags)}")

    print("\nCorrelation Theorem Verification:")
    print(f"numpy vs scipy max diff: {np.max(np.abs(acf_np - acf_sp)):.2e}")
    print(f"numpy vs IFFT max diff: {np.max(np.abs(acf_np - acf_freq)):.2e}")
    print(f"scipy vs IFFT max diff: {np.max(np.abs(acf_sp - acf_freq)):.2e}")
    print(f"Relative error (numpy vs IFFT): {np.max(np.abs(acf_np - acf_freq)) / np.max(np.abs(acf_np)) * 100:.4f}%")

    return {
        'acf_np': acf_np,
        'acf_sp': acf_sp,
        'acf_freq': acf_freq,
        'lags': lags
    }


# In[5]:


def plot_spectral_leakage_multiview(data_entry, freq_dense_points=20001, noise_floor=1e-7, 
                                   use_positive_freq=True, statistics=False):
    """
    Plot spectral leakage with multiple zoom levels in subplots, automatically centering on the dominant peak.
    """
    fs = data_entry['sample_rate']
    block = data_entry['data'][0]
    N = len(block)

    # Check if data is complex
    is_complex = np.iscomplexobj(block)

    # Measured FFT / Power
    spec = np.fft.fftshift(np.fft.fft(block))
    freq = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    power = np.abs(spec)**2

    # Find dominant peak 
    if use_positive_freq and not is_complex:
        # For real data, focus on positive frequencies
        pos_mask = freq >= 0
        pos_freq = freq[pos_mask]
        pos_power = power[pos_mask]
        f0_idx = np.argmax(pos_power)
        f0 = pos_freq[f0_idx]
    else:
        # Use the absolute maximum
        f0 = freq[np.argmax(power)]

    # Theoretical envelope 
    def rect_window_envelope_sq(freqs):
        delta_f = freqs - f0
        x = N * delta_f / fs
        env = (N * np.sinc(x))**2
        env = np.maximum(env, noise_floor)
        return env

    # Compute envelopes
    freq_dense = np.linspace(freq.min(), freq.max(), freq_dense_points)
    env_dense = rect_window_envelope_sq(freq_dense)

    # Scale envelopes
    max_power = np.max(pos_power) if (use_positive_freq and not is_complex) else np.max(power)
    env_dense *= max_power / np.max(env_dense)

    # Clip for plotting
    power_clipped = np.maximum(power, noise_floor)

    # Create subplots 
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Define zoom ranges centered on f0
    f0_kHz = f0 / 1e3
    zoom_ranges = [
        None,                              # Full spectrum
        (f0_kHz - 100, f0_kHz + 100),     # ±100 kHz around peak
        (f0_kHz - 20, f0_kHz + 20)        # ±20 kHz around peak
    ]
    titles = [
        'Full Spectrum', 
        f'Zoomed: ±100 kHz around peak ({f0_kHz:.1f} kHz)', 
        f'Detail: ±20 kHz around peak ({f0_kHz:.1f} kHz)'
    ]

    for ax, zoom, title in zip(axes, zoom_ranges, titles):
        ax.semilogy(freq/1e3, power_clipped, label='Measured (DFT)', 
                   alpha=0.8, linewidth=1.5)
        ax.semilogy(freq_dense/1e3, env_dense, 'r--', lw=1.5, 
                   label='Envelope (dense)', alpha=0.6)

        # Add vertical line at peak
        ax.axvline(f0_kHz, color='green', linestyle='--', alpha=0.5, 
                  label=f'Peak at {f0_kHz:.1f} kHz')

        # For real data, also mark the negative frequency peak
        if not is_complex and use_positive_freq:
            ax.axvline(-f0_kHz, color='green', linestyle=':', alpha=0.3, 
                      label=f'Mirror at {-f0_kHz:.1f} kHz')

        if zoom is not None:
            ax.set_xlim(zoom)

        ax.set_ylim(noise_floor, np.max(power) * 10)
        ax.set_xlabel('Frequency (kHz)', fontsize=12)
        ax.set_ylabel('Power (log scale)', fontsize=12)
        ax.set_title(f'Spectral Leakage: {title}', fontsize=13)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.show()

    if statistics:
        # Print statistics
        print(f"\nSpectral Leakage Analysis:")
        print(f"Data type: {'Complex (I/Q)' if is_complex else 'Real'}")
        print(f"Analyzing peak at: {f0/1e3:.2f} kHz")
        print(f"Sample rate: {fs/1e6:.2f} MHz")
        print(f"Nyquist frequency: ±{fs/2e6:.2f} MHz")
        print(f"Frequency resolution (Δf = fs/N): {fs/N/1e3:.3f} kHz")
        print(f"First null expected at: ±{fs/N/1e3:.3f} kHz from peak")


# In[6]:


def nyquist_windows_plot(entry, W=4, Nfreq=8*2048, use_block=0, plot_grid=(2,4)):
    """
    Compute DFT over an extended frequency axis of ±W * fs/2 and plot each Nyquist window
    in a 2x4 grid (or specified plot_grid). This uses a direct DFT sampling method over the freq grid.
    - entry: data_dict item (must have sample_rate)
    - W: number of half-windows (windows extend from -W*fs/2 .. +W*fs/2)
    - Nfreq: total number of frequency points in the extended DFT
    - use_block: index of block to analyze
    """
    fs = entry.get('sample_rate')
    if fs is None:
        raise ValueError("sample_rate missing in entry; supply fs or ensure metadata present")

    block = entry['data'][use_block]
    N = len(block)

    # center time around zero for DFT formulation
    t = (np.arange(N) - N//2) / fs

    # extended freq axis
    f_min = -W * fs/2
    f_max =  W * fs/2
    freqs = np.linspace(f_min, f_max, Nfreq)

    # compute DFT in chunks to save memory
    def dft_at_freqs(x, t, freqs, chunk=4096):
        out = np.empty(len(freqs), dtype=complex)
        for i in range(0, len(freqs), chunk):
            f_chunk = freqs[i:i+chunk]
            out[i:i+len(f_chunk)] = np.exp(-2j*np.pi*np.outer(f_chunk, t)).dot(x)
        return out

    # subtract mean to reduce DC contamination
    x = block - np.mean(block)
    spec = dft_at_freqs(x, t, freqs, chunk=2048)
    power = np.abs(spec)**2

    # Split into contiguous Nyquist windows: there are 2*W windows across -W..+W-1
    num_windows = 2 * W
    # we want to plot num_windows windows in plot_grid (2x4)
    window_edges = np.linspace(f_min, f_max, num_windows+1)

    # prepare grid
    rows, cols = plot_grid
    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), sharex=False)
    axs = axs.flatten()
    for k in range(num_windows):
        lo = window_edges[k]
        hi = window_edges[k+1]
        mask = (freqs >= lo) & (freqs < hi)
        ax = axs[k]
        ax.semilogy(freqs[mask]/1e3, power[mask], lw=0.8)
        ax.set_title(f"Nyquist window {k - W}: {lo/1e3:.0f}–{hi/1e3:.0f} kHz")
        ax.grid(True, which='both', ls=':', alpha=0.6)
        ax.set_ylabel('Power (arb.)')
    for ax in axs[num_windows:]:
        ax.axis('off')
    axs[-1].set_xlabel('Frequency (kHz)')
    plt.tight_layout()
    plt.show()
    return {'freqs': freqs, 'power': power, 'window_edges': window_edges}


# In[ ]:




