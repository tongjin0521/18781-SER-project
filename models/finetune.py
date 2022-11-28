# - Finetune flow model

import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import Wav2Vec2Processor, WavLMModel
from transformers.modeling_outputs import CausalLMOutput
from torch import nn
import pdb


class WavLMForCTCnCLS(nn.Module):

    def __init__(self, config, cls_len=4, alpha=0.01):
        super(WavLMForCTCnCLS, self).__init__()
        self.config = config
        self.wavLM = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
        self.dropout = nn.Dropout(config.final_dropout)
        self.cls_head = nn.Linear(768, cls_len)

    def freeze_feature_extractor(self):
        self.wavLM.feature_extractor._freeze_parameters()

    def model_init(self):
        self.cls_head.reset_parameters()
        
    def forward(self, inputs):
        outputs = self.wavLM(**inputs)
        hidden_states = outputs[0] # this is the last layer's hidden states
        hidden_states = self.dropout(hidden_states)
        # - L x 768 (mean)-> 1 x 768 (Linear)-> 1 x 4
        res = self.cls_head(torch.mean(hidden_states, dim=1))
        
        return res