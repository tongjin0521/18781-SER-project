import librosa
import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
import json
import csv

class HandcraftedFeatureExtractor:
    def __init__(self):
        pass
    
    def forward(self, wav):
        sig_mean = np.mean(abs(wav))
        sig_std = np.std(wav)

        rmse = librosa.feature.rms(y = wav + 0.0001)[0] 
        rmse_mean = np.mean(rmse)
        rmse_std = np.std(rmse)

        silence = 0
        for e in rmse:
            if e <= 0.4 * np.mean(rmse):
                silence += 1
        silence /= float(len(rmse))

        wav_harmonic = librosa.effects.hpss(wav)[0]
        harmonic = np.mean(wav_harmonic) * 1000 # harmonic (scaled by 1000)

        cl = 0.45 * sig_mean
        center_clipped = []
        for s in wav:
            if s >= cl:
                center_clipped.append(s - cl)
            elif s <= -cl:
                center_clipped.append(s + cl)
            elif np.abs(s) < cl:
                center_clipped.append(0)
        auto_corrs = librosa.core.autocorrelate(np.array(center_clipped))
        auto_corr_max = 1000 * np.max(auto_corrs)/len(auto_corrs) # auto_corr_max (scaled by 1000)
        auto_corr_std = np.std(auto_corrs)
        
        res = [sig_mean,sig_std,rmse_mean,rmse_std,silence,harmonic,auto_corr_max,auto_corr_std]
        
        return [str(res_i) for res_i in res]
    
    def process_wav(self, file_path):
        y, _sr = librosa.load(file_path)
        
        # - following is same as forward

        raw_data = y

        sig_mean = np.mean(abs(raw_data))
        sig_std = np.std(raw_data)

        rmse = librosa.feature.rms(y = raw_data + 0.0001)[0] 
        rmse_mean = np.mean(rmse)
        rmse_std = np.std(rmse)

        silence = 0
        for e in rmse:
            if e <= 0.4 * np.mean(rmse):
                silence += 1
        silence /= float(len(rmse))

        raw_data_harmonic = librosa.effects.hpss(raw_data)[0]
        harmonic = np.mean(raw_data_harmonic) * 1000 # harmonic (scaled by 1000)

        cl = 0.45 * sig_mean
        center_clipped = []
        for s in raw_data:
            if s >= cl:
                center_clipped.append(s - cl)
            elif s <= -cl:
                center_clipped.append(s + cl)
            elif np.abs(s) < cl:
                center_clipped.append(0)
        auto_corrs = librosa.core.autocorrelate(np.array(center_clipped))
        auto_corr_max = 1000 * np.max(auto_corrs)/len(auto_corrs) # auto_corr_max (scaled by 1000)
        auto_corr_std = np.std(auto_corrs)
        
        res = [sig_mean,sig_std,rmse_mean,rmse_std,silence,harmonic,auto_corr_max,auto_corr_std]
        return [str(res_i) for res_i in res]


# - If you want to extract the handcrafted features in this file, use following code.
'''
def main(wav_dir_path,res_file_path):
    all_wav_files = os.listdir(wav_dir_path)
   
    res_data = {}
    for wav_file_i in tqdm(all_wav_files):
        try:
            res_data[wav_file_i] = process_wav(wav_dir_path+wav_file_i)
        except:
            print('An exception occured for {}'.format(wav_file_i))
    # print(res_data)
    with open(res_file_path, 'w') as f:
        json.dump(res_data, f, indent=4)

if __name__ == '__main__':
    wav_dir_path = "/jet/home/tjin1/wlchiu/IEMOCAP_full_release/Dataset/IEMOCAP/Audio_16k/"
    res_file_path = "handcrafted_features.json"
    main(wav_dir_path,res_file_path)
'''
