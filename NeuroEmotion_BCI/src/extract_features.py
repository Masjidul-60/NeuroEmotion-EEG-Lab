import numpy as np
from scipy.signal import welch
try:
    from scipy.integrate import simpson as simps
except ImportError:  # pragma: no cover
    from scipy.integrate import simps


EEG_BANDS = {
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30),
    "Gamma": (30, 45),
}


def _prepare_signal(data):
    """Convert any EEG-like input to a 2D array shaped (channels, samples)."""
    signal = np.asarray(data, dtype=float)
    if signal.size == 0:
        raise ValueError("Input data is empty.")

    if signal.ndim == 1:
        return signal.reshape(1, -1)

    if signal.ndim >= 2:
        return signal.reshape(-1, signal.shape[-1])

    raise ValueError("Input data must be a 1D or 2D numeric array.")


def compute_bandpower(data, sf, band):
    """Calculate band power for one or many EEG channels.

    Parameters
    ----------
    data : array-like
        EEG signal with shape (samples,) or (channels, samples).
    sf : float
        Sampling frequency in Hz.
    band : tuple
        Lower and upper frequency bounds in Hz.

    Returns
    -------
    float or np.ndarray
        Power value for single-channel input, or one value per channel.
    """
    if sf <= 0:
        raise ValueError("Sampling frequency must be greater than zero.")

    low, high = band
    if low > high:
        raise ValueError("Band lower bound must not exceed upper bound.")

    channels = _prepare_signal(data)
    nperseg = min(4 * int(sf), channels.shape[1])
    nperseg = max(8, nperseg)

    result = []
    for channel in channels:
        if np.all(np.isfinite(channel)):
            freqs, psd = welch(channel, fs=sf, nperseg=min(nperseg, len(channel)))
        else:
            raise ValueError("Input data contains NaN or infinite values.")

        idx_band = np.logical_and(freqs >= low, freqs <= high)
        if not np.any(idx_band):
            result.append(0.0)
            continue

        band_power = simps(psd[idx_band], freqs[idx_band])
        result.append(float(band_power))

    result = np.asarray(result, dtype=float)
    if result.size == 1:
        return float(result[0])
    return result


def extract_all_bands(data, sf=256):
    """Extract standard EEG frequency-band features.

    Returns a dictionary mapping {'Theta', 'Alpha', 'Beta', 'Gamma'} to their
    band-power values.  For multi-channel input, the values are arrays with one
    entry per channel.
    """
    features = {}
    for band_name, limits in EEG_BANDS.items():
        features[band_name] = compute_bandpower(data, sf, limits)
    return features


def extract_emotion_features(data, sf=256):
    """Return EEG power features plus derived ratios useful for emotion analysis."""
    bands = extract_all_bands(data, sf)
    alpha = np.asarray(bands["Alpha"], dtype=float)
    beta = np.asarray(bands["Beta"], dtype=float)
    theta = np.asarray(bands["Theta"], dtype=float)
    gamma = np.asarray(bands["Gamma"], dtype=float)

    features = {
        **bands,
        "alpha_beta_ratio": np.divide(alpha, beta, out=np.zeros_like(alpha, dtype=float), where=beta != 0),
        "theta_alpha_ratio": np.divide(theta, alpha, out=np.zeros_like(theta, dtype=float), where=alpha != 0),
        "gamma_beta_ratio": np.divide(gamma, beta, out=np.zeros_like(gamma, dtype=float), where=beta != 0),
    }
    return features


def classify_emotion(data, sf=256):
    """Simple heuristic classifier: alpha-dominant -> relaxed, beta-dominant -> focused."""
    features = extract_all_bands(data, sf)
    alpha = np.asarray(features["Alpha"], dtype=float)
    beta = np.asarray(features["Beta"], dtype=float)

    if np.ndim(alpha) == 0:
        return "Relaxed" if float(alpha) > float(beta) else "Focused"

    labels = np.where(alpha > beta, "Relaxed", "Focused")
    return labels.tolist() if labels.size > 1 else labels.item()
