#from dataloader import CustomEmoDataset
import torchaudio # - for MFCC transform
import torch # - for torch.from_numpy

class MFCCFeatureExtractor:
    def __init__(self):
        self.T = torchaudio.transforms.MFCC()
        
    '''  
    def create_files(self):
        datadir = "inputdata/Audio_16k/"
        labeldirs = ["inputdata/labels_sess/label_1.json", "inputdata/labels_sess/label_2.json", "inputdata/labels_sess/label_3.json", "inputdata/labels_sess/label_4.json", "inputdata/labels_sess/label_5.json"]

        for session, labeldir in enumerate(labeldirs):
            mfccs = []
            dataset = CustomEmoDataset(datadir, labeldir)
            print("dataset created")
            for i in range(len(dataset.train_dataset)):
                train_wav, _ = dataset.train_dataset[0]
                mfccs.append(T(torch.from_numpy(train_wav)))
            torch.save(mfccs, "MFCC_"+str(session)+".pk")
            print("output mfcc file")
    '''

    def forward(self, wav):
        if type(wav) != torch:
            wav = torch.from_numpy(wav)
        return self.T(wav)