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

def extractHandcraftedFeature(wav):
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


def extractMFCCFeature(wav, sample_rate = 16000):
    transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
    )
    return transform(torch.from_numpy(wav))

# - We don't use this function any more
def create_loader(
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
    
    dev_dataset = EmoDataset(sorted_data, params, is_train = is_train, is_aug = is_aug)
    
    train_len = int((1 - valid_ratio) * len(dev_dataset))
    train_valid_lens = [train_len, len(dev_dataset) - train_len]
    
    train_dataset, valid_dataset = random_split(dev_dataset, train_valid_lens)
    
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

    return train_dataset, train_loader, valid_dataset, valid_loader

# - five fold seperations, our standard data preparation method, no batch
def create_loader_five_fold(
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
    
    sessions = [[] for _ in range(5)]
    for sentence in sorted_data:
        sessions[int(sentence[0][4])-1].append(sentence)
    
    # - group by session
    test_sessions  = []
    train_sessions = []
    for test_id in range(5):
        test_sessions.append(sessions[test_id])
        temp = []
        for session_id in range(5):
            if session_id != test_id:
                temp += sessions[session_id]
        train_sessions.append(temp)

    
    train_datasets, train_loaders, valid_datasets, valid_loaders, test_datasets, test_loaders = [], [], [], [], [], []
    for fold_id in range(5):
        test_dataset = EmoDataset(test_sessions[fold_id], params, is_train = is_train, is_aug = is_aug)
        dev_dataset = EmoDataset(train_sessions[fold_id], params, is_train = is_train, is_aug = is_aug)
        
        train_len = int((1 - valid_ratio) * len(dev_dataset))
        train_valid_lens = [train_len, len(dev_dataset) - train_len]
        
        train_dataset, valid_dataset = random_split(dev_dataset, train_valid_lens)
    
        test_loader = DataLoader(
            dataset=test_dataset,
            num_workers=params.nworkers,
            batch_size=params.batch_size,
            pin_memory=True,
        )
        
        train_loader = DataLoader(
            dataset=train_dataset,
            num_workers=params.nworkers,
            batch_size=params.batch_size,
            pin_memory=True,
        )
        
        valid_loader = DataLoader(
            dataset=valid_dataset,
            num_workers=params.nworkers,
            batch_size=params.batch_size,
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
    lambda_key: str = "embedding_dim",
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
        # Check for min_batch_size and fixes the batches if needed
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

# - old dataset definition, for feature extraction only
class CustomEmoDataset:
    def __init__(self, datadir, labeldir, maxseqlen=12):
        super().__init__()
        self.maxseqlen = maxseqlen * 16000 
        with open(labeldir, 'r') as f:
            self.label = json.load(f) 
        self.emoset = list(set([emo for split in self.label.values() for emo in split.values()]))
        self.emoset = list(sorted(self.emoset))
        self.nemos = len(self.emoset)
        self.train_dataset = _CustomEmoDataset(datadir, self.label['Train'], self.emoset, 'training')
        if self.label['Val']:
            self.val_dataset = _CustomEmoDataset(datadir, self.label['Val'], self.emoset, 'validation')
        if self.label['Test']:
            self.test_dataset = _CustomEmoDataset(datadir, self.label['Test'], self.emoset, 'testing')

    def seqCollate(self, batch):
        getlen = lambda x: x[0].shape[0]
        max_seqlen = max(map(getlen, batch))
        target_seqlen = min(self.maxseqlen, max_seqlen)
        def trunc(x):
            x = list(x)
            if x[0].shape[0] >= target_seqlen:
                x[0] = x[0][:target_seqlen]
                output_length = target_seqlen
            else:
                output_length = x[0].shape[0]
                over = target_seqlen - x[0].shape[0]
                x[0] = np.pad(x[0], [0, over])
            ret = (x[0], output_length, x[1])
            return ret
        batch = list(map(trunc, batch))
        return default_collate(batch)

# - old dataset definition, for feature extraction only
class _CustomEmoDataset(data.Dataset):
    def __init__(self, datadir, label, emoset,
                 split, maxseqlen=12):
        super().__init__()
        self.maxseqlen = maxseqlen * 16000 #Assume sample rate of 16000
        self.split = split
        self.label = label #{wavname: emotion_label}
        self.emos = Counter([self.label[n] for n in self.label.keys()])
        self.emoset = emoset
        self.labeldict = {k: i for i, k in enumerate(self.emoset)}
        self.datasetbase = list(self.label.keys())
        self.dataset = [os.path.join(datadir, x) for x in self.datasetbase]

        #Print statistics:
        print (f'Statistics of {self.split} splits:')
        print ('----Involved Emotions----')
        for k, v in self.emos.items():
            print (f'{k}: {v} examples')
        l = len(self.dataset)
        print (f'Total {l} examples')
        print ('----Examples Involved----\n')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        dataname = self.dataset[i]
        wav, _sr = sf.read(dataname)
        _label = self.label[self.datasetbase[i]]
        label = self.labeldict[_label]
        wav = wav.astype(np.float32)
        hfeat, mfeat, cfeat = None, None, None
        hfeat = extractHandcraftedFeature(wav)
        mfeat = extractMFCCFeature(wav)
        #cfeat = self.C.forward(wav)
        
        return dataname, wav, hfeat, mfeat, cfeat, label
    
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
        #params.batch_enable
        
    def __len__(self):
        """Returns the number of examples in the dataset"""
        return len(self.data)
    
    # - the file structure I prepared for the batch enable version is slightly different
    def __getitem__(self, idx: str):
        if self.batch_enable:
            embedding_file = self.data[idx]["input"]["embedding"]
            embedding_len = self.data[idx]["input"]["embedding_dim"][0]
        else:
            embedding_file = self.data[idx][1]["input"]["embedding"]
            embedding_len = self.data[idx][1]["input"]["embedding_dim"][0]
            
        with open(embedding_file, 'rb') as f:
            embedding = pickle.load(f)
        audio_feat = embedding
        feat_len = embedding_len
        
        if self.batch_enable:
            target = self.data[idx]["output"] if self.is_train else None
        else:
            target = self.data[idx][1]["output"] if self.is_train else None
            
        if target != None:
            target = torch.nn.functional.one_hot(torch.tensor(target), num_classes = 4).float()
        if self.batch_enable:
            return audio_feat, feat_len, target, idx
        return audio_feat, target
    
    def getData(self):
        return self.data
    
    # TODO: collate_function for batch
    def collate_function(self, batch):
        """Retrieves an item from the dataset given the index

        :param generator batch- Batch of data
        :returns torch.Tensor padded_feats- Speech features
        :returns torch.Tensor padded_targets- Output sequence
        :returns list data_keys- ID key
        """
 
        padded_feats = pad_list([torch.from_numpy(x[0]) for x in batch], self.pad)
        targets = torch.zeros([len(batch), 4], dtype=torch.float32)
        feat_len = [x[1] for x in batch]
        for i,x in enumerate(batch):
            targets[i] = x[2]
        # - targets = torch.stack([torch.tensor(x[1]) for x in batch])
        id_keys = [x[3] for x in batch]
        return padded_feats, feat_len, targets, id_keys
    
# [B * (Total_data)] b=1, ..., B
#
