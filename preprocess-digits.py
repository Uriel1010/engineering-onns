import os
import librosa
import math
import warnings
import numpy as np

warnings.filterwarnings("ignore")

DATASET_PATH = "free-spoken-digit-dataset/recordings"
NPZ_PATH = "free-spoken-digit-dataset/digitsData.npz"

SAMPLE_RATE = 22050
DURATION = 1  # in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc_to_npz(dataset_path, npz_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """
    Extracts MFCCs from the free spoken digit dataset and saves them into an NPZ file along with the digit labels.

    :param dataset_path: str, Path to the free spoken digit dataset directory
    :param npz_path: str, Path to the NPZ file used to save MFCCs, labels, and mapping
    :param num_mfcc: int, Number of coefficients to extract, default is 13
    :param n_fft: int, Interval to consider when applying FFT, measured in number of samples, default is 2048
    :param hop_length: int, Sliding window for FFT, measured in number of samples, default is 512
    :param num_segments: int, Number of segments to divide sample tracks into, default is 5
    :return: None
    """
    # data lists to store mapping, labels, and MFCCs
    mapping = []
    labels = []
    mfcc_data = []

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        for f in filenames:
            if f.endswith(".wav"):
                if int(f[0]) not in mapping:
                    mapping.append(int(f[0]))
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                for d in range(num_segments):
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        mfcc_data.append(mfcc.tolist())
                        labels.append(int(f[0]))
                        print("{}, segment:{}".format(file_path, d + 1))

    # save MFCCs and labels to npz file
    np.savez(npz_path, mfcc=mfcc_data, labels=labels, mapping=mapping)

def load_data(npz_path):
    with np.load(npz_path) as data:
        mfcc = data['mfcc']
        labels = data['labels']
    return mfcc, labels

if __name__ == "__main__":
    save_mfcc_to_npz(DATASET_PATH, NPZ_PATH, num_segments=10)
    mfcc_data, labels = load_data(NPZ_PATH)
    pass
