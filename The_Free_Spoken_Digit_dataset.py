import os
import numpy as np
from scipy.io.wavfile import read
from sklearn.model_selection import train_test_split

def load_data(path):
    x_data = []
    y_data = []
    x_max_len = 18000
    for digit_folder in os.listdir(path):
        digit_path = os.path.join(path, digit_folder)
        if digit_folder.endswith('.wav'):
            sr, data = read(digit_path)
            fft_data = np.fft.fft(data)
            if len(data)>=x_max_len:
                data = data[:x_max_len]
            else:
                data = np.pad(data,(0,x_max_len-len(data)) , mode='constant', constant_values=0)/max(data)
            x_data.append(data)
            y_data.append(int(digit_folder[0]))

    return x_data, y_data