import librosa
import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
import json
import csv

#########################
# Augmentation methods
#########################
def noise(data):
    """
    Adding White Noise.
    """
    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = 0.01*np.random.uniform()*np.amax(data)   # more noise reduce the value to 0.5
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data
    
def shift(data):
    """
    Random Shifting.
    """
    s_range = int(np.random.uniform(low=-5, high = 5)*1000)  #default at 500
    return np.roll(data, s_range)
    
def dyn_change(data):
    """
    Random Value Change.
    """
    dyn_change = np.random.uniform(low=-0.5 ,high=7)  # default low = 1.5, high = 3
    return (data * dyn_change)

def main(wav_dir_path):
    all_wav_files = os.listdir(wav_dir_path)
   
    for wav_file_i in all_wav_files:
        if wav_file_i.find("aug_aug") ==-1:
            file_path = wav_dir_path +wav_file_i
            try:
                new_wav_file = wav_file_i.split(".")[0] + "_aug.wav"
                raw_data, sr = librosa.load(file_path)
                noised_data = noise(raw_data)
                shifted_noised_data = shift(noised_data)
                final_data = dyn_change(shifted_noised_data)
                sf.write(wav_dir_path+new_wav_file, final_data, sr)
                print("Success on " + new_wav_file)
            except:
                print('An exception occured for {}'.format(wav_file_i))



if __name__ == '__main__':
    wav_dir_path = "/ocean/projects/tra220029p/tjin1/term_project/inputdata/Audio_16k/"
    main(wav_dir_path)
