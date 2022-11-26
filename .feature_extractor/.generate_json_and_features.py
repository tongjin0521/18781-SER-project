# - create json and data files

import sys
import os
import configargparse
import random
import torch
import numpy as np
from dataloader import CustomEmoDataset
import json

def main(cmd_args):
    datadir = "inputdata/Audio_16k/"
    labeldirs = ["inputdata/labels_sess/label_1.json", "inputdata/labels_sess/label_2.json", "inputdata/labels_sess/label_3.json", "inputdata/labels_sess/label_4.json", "inputdata/labels_sess/label_5.json"]
    D = {}
    for session, labeldir in enumerate(labeldirs):
        dataset = CustomEmoDataset(datadir, labeldir)
        print(f'processing {labeldir}')
        for index in range(len(dataset.train_dataset)):
            if index % 500 == 0:
                print(f'processing {index}-th file...')
            path, wav, hfeat, mfccfeat, cfeat, label = dataset.train_dataset[index]
            
            
            # - path analysis
            path = path.split('/')
            root , _, filename = path
            id = filename[:-4]
            if id in D:
                continue
            
            # - check gender
            gender = "Male"
            if filename[5] == "F":
                gender = "Female"
            
            # - paths
            wav_path = root +"/wav/" + id + ".wav"
            hpath = root +"/handcraft/" + id + ".pk"
            mfccpath = root +"/mfcc/" + id + ".pk"
            embeddingpath = root + "/embedding/" + id + ".pk"
            
            # - data type casting
            hfeat = np.array(hfeat).astype(np.float32) # - str list to float32 np array
            mfccfeat = mfccfeat.numpy() # - torch to float32 np array
            
            # - save file
            if not os.path.exists(wav_path):
                torch.save(wav, wav_path)
            if not os.path.exists(hpath):
                torch.save(hfeat, hpath)
            if not os.path.exists(mfccpath):
                torch.save(mfccfeat, mfccpath)
        
            # cfeat, TODO
        
            # - writing in dictionary
            D[id] = {"id":id,
                    "gender": gender,
                    "input":{"wav": wav_path, 
                            "wav_dim": wav.shape, 
                            "handcraft": hpath,
                            "handcraft_dim": hfeat.shape,
                            "mfcc": mfccpath, 
                            "mfcc_dim": mfccfeat.shape,
                            "embedding": embeddingpath,
                            "embedding_dim": -1
                                },
                    "output":label}

    # - dump json
        json_object = json.dumps(D, indent=4)
        with open("sample.json", "w") as outfile:
            outfile.write(json_object)

if __name__ == "__main__":
    main(sys.argv[1:])