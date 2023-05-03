import os
import sys
import glob
import librosa
from scipy.io import wavfile
import warnings
import numpy as np
from tqdm import tqdm


DATASET_PATH = "free-spoken-digit-dataset/recordings"
NPZ_PATH = "free-spoken-digit-dataset/digits_mfcc.npz"


def save_mfcc_to_npz(dataset_path, npz_path, num_mfcc=20, hop_length=128, n_fft=512):
    """
    Extracts MFCCs from the free spoken digit dataset and saves them into an NPZ file along with the digit labels.

    :param dataset_path: str, Path to the free spoken digit dataset directory
    :param npz_path: str, Path to the NPZ file used to save MFCCs, labels, and mapping
    :param num_mfcc: int, Number of coefficients to extract, default is 13
    :param hop_length: int, Sliding window for FFT, measured in number of samples, default is 512
    :return: None
    """
    # data lists to store MFCCs with labels
    mfcc_data = []
    labels = []

    signals = []
    max_length = 0
    max_sample_rate = 0
    files = glob.glob(os.path.join(dataset_path, '*.wav'))
    for f in files:
        sample_rate, signal = wavfile.read(f)
        labels.append(os.path.basename(f))
        signals.append(signal)
        if signal.shape[0] > max_length:
            max_length = signal.shape[0]
        if sample_rate > max_sample_rate:
            max_sample_rate = sample_rate

    with tqdm(total=len(files)) as pbar:
        for signal, label in zip(signals, labels):
            pbar.set_description(f'Processing recording {label}')
            signal = np.pad(signal, (0, max_length - signal.shape[0]))
            mfcc = librosa.feature.mfcc(
                y=signal.astype(np.double),
                sr=max_sample_rate,
                n_mfcc=num_mfcc,
                hop_length=hop_length,
                n_fft=n_fft
            )
            mfcc_data.append(mfcc.T)
            pbar.update(1)

    # # save MFCCs and labels to npz file
    np.savez(npz_path, mfcc=mfcc_data, labels=labels)


def load_data(npz_path):
    with np.load(npz_path) as data:
        mfcc = data['mfcc']
        labels = data['labels']
    return mfcc, labels


if __name__ == "__main__":
    save_mfcc_to_npz(DATASET_PATH, NPZ_PATH)
    mfcc_data, labels = load_data(NPZ_PATH)
