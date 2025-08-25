import mne

def load_eeg(file_path: str):
    """Load EEG data from .edf or .set files using MNE."""
    try:
        if file_path.endswith('.edf'):
            raw = mne.io.read_raw_edf(file_path, preload=True)
        elif file_path.endswith('.set'):
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
        elif file_path.endswith('.fif'):
            raw = mne.io.read_raw_fif(file_path, preload=True)
        else:
            raise ValueError('Unsupported EEG file format: ' + file_path)
        return raw
    except Exception as e:
        print(f"Error loading EEG: {e}")
        return None

def preprocess_eeg(raw, l_freq: float = 1.0, h_freq: float = 40.0):
    """Basic EEG preprocessing: band-pass filter and return Raw object."""
    if raw is None:
        return None
    raw.filter(l_freq, h_freq, fir_design='firwin')
    return raw
