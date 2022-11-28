import os
from torch.utils import data
import torch
import json
import numpy as np
from collections import Counter
import soundfile as sf
from torch.utils.data.dataloader import default_collate
import kaldiio
import configargparse
import random
import torchaudio # - for MFCC transform
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import random_split
import librosa
import pickle
from utils import pad_list
import pdb
from transformers import Wav2Vec2Processor, WavLMModel



# - five fold seperations, our standard data preparation method, no batch
def create_loader_with_folds(
    data: dict,
    params: configargparse.Namespace,
    is_train: bool,
    min_batch_size: int = 1,
    shortest_first: bool = False,
    is_aug: bool = False,
    lambda_key: str = "wav_dim",
    valid_ratio: float = 0.2,
):
    sorted_data = sorted(
        data.items(),
        key=lambda data: int(data[1]["input"][lambda_key][0]),
        reverse=not shortest_first,
    )
    length = len(sorted_data)
    idim = int(sorted_data[0][1]["input"][lambda_key][0]) 
    odim = 1
    
    num_folds = params.fold

    sessions = [[] for _ in range(num_folds)]
    for sentence in sorted_data:
        if num_folds == 5:
            sessions[int(sentence[0][4])-1].append(sentence)
        else:
            gender = 0
            if sentence[0][5] == 'F':
                gender = 1
            index = (int(sentence[0][4])-1)*2 + gender
            sessions[index].append(sentence)
    
    
    # - group by session
    test_sessions  = []
    train_sessions = []
    for test_id in range(num_folds):
        test_sessions.append(sessions[test_id])
        temp = []
        for session_id in range(num_folds):
            if session_id != test_id:
                temp += sessions[session_id]
        train_sessions.append(temp)

    
    train_datasets, train_loaders, valid_datasets, valid_loaders, test_datasets, test_loaders = [], [], [], [], [], []
    for fold_id in range(num_folds):
        test_dataset = EmoDataset(test_sessions[fold_id], params, is_train = is_train, is_aug = is_aug)
        dev_dataset = EmoDataset(train_sessions[fold_id], params, is_train = is_train, is_aug = is_aug)
        
        train_len = int((1 - valid_ratio) * len(dev_dataset))
        train_valid_lens = [train_len, len(dev_dataset) - train_len]
        
        train_dataset, valid_dataset = random_split(dev_dataset, train_valid_lens)
    
        test_loader = DataLoader(
            dataset=test_dataset,
            num_workers=params.nworkers,
            pin_memory=True,
        )
        
        train_loader = DataLoader(
            dataset=train_dataset,
            num_workers=params.nworkers,
            pin_memory=True,
        )
        
        valid_loader = DataLoader(
            dataset=valid_dataset,
            num_workers=params.nworkers,
            pin_memory=True,
        )
            
        
        train_datasets.append(train_dataset)
        train_loaders.append(train_loader)
        valid_datasets.append(valid_dataset)
        valid_loaders.append(valid_loader)
        test_datasets.append(test_dataset)
        test_loaders.append(test_loader)
    
    return train_datasets, train_loaders, valid_datasets, valid_loaders, test_datasets, test_loaders
    
# - try to implement the batch but still have some problems, the collate_function
def _create_loader(
    data: dict, 
    params: configargparse.Namespace,
    is_train: bool = True,
    min_batch_size: int = 1,
    shortest_first: bool = False,
    lambda_key: str = "wav_dim",
    is_aug: bool = False,
):
    """Creates batches with different batch sizes which maximizes the number of bins up to `batch_bins`.

    :param Dict[str, Dict[str, Any]] data: dictionary loaded from data.json
    :param int batch_bins: Maximum frames of a batch
    :param int min_batch_size: minimum batch size (for multi-gpu)
    :param bool shortest_first: Sort from batch with shortest samples to longest if true, otherwise reverse
    :returns: List[Tuple[str, Dict[str, List[Dict[str, Any]]]] list of batches
    """
    batch_bins = params.batch_bins

    sorted_data = sorted(
        data.items(),
        key=lambda data: int(data[1]["input"][lambda_key][0]),
        reverse=not shortest_first,
    )
    if batch_bins <= 0:
        raise ValueError(f"invalid batch_bins={batch_bins}")
    length = len(sorted_data)
    idim = int(sorted_data[0][1]["input"][lambda_key][0])
    odim = 4
    minibatches = []
    start = 0
    n = 0
    while True:
        # Dynamic batch size depending on size of samples
        b = 0
        next_size = 0
        max_olen = 0
        while next_size < batch_bins and (start + b) < length and b < params.batch_size:
            ilen = int(sorted_data[start + b][1]["input"][lambda_key][1]) * idim
            olen = 4
            if olen > max_olen:
                max_olen = olen
            next_size = (max_olen + ilen) * (b + 1)
            if next_size <= batch_bins:
                b += 1
            elif next_size == 0:
                raise ValueError(
                    f"Can't fit one sample in batch_bins ({batch_bins}): "
                    f"Please increase the value"
                )
        end = min(length, start + max(min_batch_size, b))
        batch = [element[0] for element in sorted_data[start:end]]
        if shortest_first:
            batch.reverse()
        minibatches.append(batch)
        i = -1
        while len(minibatches[i]) < min_batch_size:
            missing = min_batch_size - len(minibatches[i])
            if -i == len(minibatches):
                minibatches[i + 1].extend(minibatches[i])
                minibatches = minibatches[1:]
                break
            else:
                minibatches[i].extend(minibatches[i - 1][:missing])
                minibatches[i - 1] = minibatches[i - 1][missing:]
                i -= 1
        if end == length:
            break
        start = end
        n += 1
    lengths = [len(x) for x in minibatches]
    print(
        "[info] #Utts: {} | Created {} minibatches containing {} to {} samples, and on average {} samples".format(
            len(sorted_data),
            len(minibatches),
            min(lengths),
            max(lengths),
            int(np.mean(lengths)),
        )
    )

    minibatches = MiniBatchSampler(minibatches, shuffle=False)
    
    dataset = EmoDataset(data, params, is_train = is_train, is_aug = is_aug)

    loader = DataLoader(
        dataset=dataset,
        batch_sampler=minibatches,
        num_workers=params.nworkers,
        collate_fn=dataset.collate_function,
        pin_memory=True,
    )

    return dataset, loader, minibatches


class MiniBatchSampler:
    def __init__(self, batches, shuffle):
        super(MiniBatchSampler, self).__init__()
        self.batches = batches
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
            print(f"[info] Minibatches have been randomly shuffled.")
        for bt in self.batches:
            yield bt
    
# - our dataset
class EmoDataset(data.Dataset):
    def __init__(self, data: dict, params: configargparse.Namespace, is_train: bool = False, is_aug: bool = False):
        self.data = data
        self.params = params
        self.is_train = is_train
        self.is_aug = is_aug
        self.pad = params.pad
        self.target_pad = params.target_pad
        self.batch_enable = params.batch_enable
        self.handcrafted_features = params.handcrafted_features
        self.processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
        
    def __len__(self):
        """Returns the number of examples in the dataset"""
        return len(self.data)
    
    # - the file structure I prepared for the batch enable version is slightly different
    # - TODO: fix the file difference and make interface clean
    def __getitem__(self, idx: str):
        if self.batch_enable:
            wav_file = self.data[idx]["input"]["wav"]
            wav_len = self.data[idx]["input"]["wav_dim"][0]
            handcrafted_features_file = self.data[idx]["input"]["handcraft"]
            target = self.data[idx]["output"] if self.is_train else None
        else:
            wav_file = self.data[idx][1]["input"]["wav"]
            wav_len = self.data[idx][1]["input"]["wav_dim"][0]
            handcrafted_features_file = self.data[idx][1]["input"]["handcraft"]
            target = self.data[idx][1]["output"] if self.is_train else None
        
        # - note that wav file is save by torch.save
        wav = torch.load(wav_file)
        audio_feat = wav
        feat_len = wav_len
        
        handcrafted_features = -1
        if self.handcrafted_features:
            handcrafted_features = torch.load(handcrafted_features_file)
            handcrafted_features = np.append(handcrafted_features, np.array([float(handcrafted_features_file.split("/")[-1][5] == 'F')]).astype(np.float32))
        
        if target != None:
            target = torch.nn.functional.one_hot(torch.tensor(target), num_classes = 4).float()
            
        audio_feat = audio_feat.reshape(-1)
        print(f'type of audio feature {type(audio_feat)}')
        print(f'shape of audio feature {audio_feat.shape}')
        print(f'value of audio feature {audio_feat}')
        audio_feat = self.processor(audio_feat, sampling_rate=16000, return_tensors="pt")
        print(f'type of processed feature {type(audio_feat)}')
        audio_feat['input_values'] = audio_feat['input_values'].reshape(-1)
        s = audio_feat['input_values'].shape
        print(f'shape of processedo feature {s}')
        print(f'value of processedo feature {audio_feat}')
        return audio_feat, feat_len, handcrafted_features, target, idx
    
    def getData(self):
        return self.data
    
    def collate_function(self, batch):
        """Retrieves an item from the dataset given the index

        :param generator batch- Batch of data
        :returns torch.Tensor padded_feats- Speech features
        :returns torch.Tensor padded_targets- Output sequence
        :returns list data_keys- ID key
        """
 
        padded_feats = pad_list([torch.from_numpy(x[0]) for x in batch], self.pad)
        feat_len = [x[1] for x in batch]
        # - [weilunc, 11/27/13:44] Fix the nparray -> tensor issue
        # handcrafted_features = torch.FloatTensor([x[2] for x in batch])
        handcrafted_features = torch.zeros([len(batch), len(batch[0][2])], dtype=torch.float32)
        targets = torch.zeros([len(batch), len(batch[0][3])], dtype=torch.float32)
        for i,x in enumerate(batch):
            handcrafted_features[i] = torch.from_numpy(x[2])
            targets[i] = x[3]
        
        id_keys = [x[4] for x in batch]
        return padded_feats, feat_len, handcrafted_features, targets, id_keys
    
