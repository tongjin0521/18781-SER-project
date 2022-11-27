from transformers import WavLMModel, WavLMPreTrainedModel
from torch import nn

# - the upstream-to-downstream model for both ASR and SER(multi-task learning)
class WavLMForCTCnCLS(WavLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.pretrain = WavLMModel(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.cls_head = nn.Linear(config.hidden_size, cls_len)
        self.init_weights()