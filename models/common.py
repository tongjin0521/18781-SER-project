# - ref: https://github.com/s3prl/s3prl/blob/a80b403d7c5f3005beef8cc1086c2fe8dea25cc9/s3prl/nn/common.py

from typing import List

import torch.nn as nn
import torch
import models.pooling as pooling


class UtteranceLevel(nn.Module):
    """
    Args:
        input_size (int): input_size
        output_size (int): output_size
        hidden_sizes (List[int]): a list of hidden layers' hidden size.
            by default is [256] to project all different input sizes to the same dimension.
            set empty list to use the vanilla single layer linear model
        activation_type (str): the activation class name in :obj:`torch.nn`. Set None to
            disable activation and the model is pure linear. Default: None
        activation_conf (dict): the arguments for initializing the activation class.
            Default: empty dict
        pooling_type (str): the pooling class name in :obj:`s3prl.nn.pooling`. Default: MeanPooling
        pooling_conf (dict): the arguments for initializing the pooling class.
            Default: empty dict
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = None,
        activation_type: str = None,
        activation_conf: dict = None,
        pooling_type: str = "MeanPooling",
        pooling_conf: dict = None,
        handcrafted_features: bool = False,
        mfcc_features: bool = False,
    ):
        super().__init__()
        self._indim = input_size
        self._outdim = output_size
        self.handcrafted_features = handcrafted_features
        self.mfcc_features = mfcc_features
        hidden_sizes = hidden_sizes or [256]

        latest_size = input_size
        hidden_layers = []
        if len(hidden_sizes) > 0:
            for size in hidden_sizes:
                hidden_layers.append(nn.Linear(latest_size, size))
                if activation_type is not None:
                    hidden_layers.append(
                        getattr(nn, activation_type)(**(activation_conf or {}))
                    )
                latest_size = size

        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        self.mfcc_layer_output_len = 0
        if self.mfcc_features:
            self.mfcc_layer_output_len = 16
            self.mfcc_layer = nn.Linear(13,self.mfcc_layer_output_len)
            self.mfcc_pooling = getattr(pooling, pooling_type)(self.mfcc_layer_output_len)

        pooling_conf = pooling_conf or {}
        self.pooling = getattr(pooling, pooling_type)(latest_size, **pooling_conf)
        if not self.handcrafted_features:
            latest_size = self.pooling.output_size
        else:
            latest_size = self.pooling.output_size + 9
            # TODO: MAKE 9 A CONSTANT / OBTAINED FROM SOMEWHERE ELSE
        
    
        latest_size += self.mfcc_layer_output_len

        self.final_proj = nn.Linear(latest_size, output_size)

    @property
    def input_size(self) -> int:
        return self._indim

    @property
    def output_size(self) -> int:
        return self._outdim

    def forward(self, x, x_len, mfcc_feat, mfcc_feat_len, handcrafted_features):
        """
        Args:
            x (torch.FloatTensor): (batch_size, seq_len, input_size)
            x_len (torch.LongTensor): (batch_size, )
        Returns:
            torch.FloatTensor
            (batch_size, output_size)
        """
        x = self.hidden_layers(x)
        x_pooled = self.pooling(x, x_len)
        if self.mfcc_features:
            mfcc_tmp = self.mfcc_layer(mfcc_feat)
            mfcc_pooled = self.pooling(mfcc_tmp,mfcc_feat_len)
        if not self.handcrafted_features:
            if not self.mfcc_features:
                y = self.final_proj(x_pooled)
            else:
                combined_data = torch.cat((x_pooled, mfcc_pooled),1)
                y = self.final_proj(combined_data)
        else:
            if not self.mfcc_features:
                combined_data = torch.cat((x_pooled, handcrafted_features),1)
                y = self.final_proj(combined_data)
            else:
                combined_data = torch.cat((x_pooled, handcrafted_features),1)
                combined_data = torch.cat((combined_data, mfcc_pooled),1)
                y = self.final_proj(combined_data)   
        return y



class Simple_FC(nn.Module):
    """
    Args:
        input_size (int): input_size
        output_size (int): output_size
        hidden_sizes (List[int]): a list of hidden layers' hidden size.
            by default is [256] to project all different input sizes to the same dimension.
            set empty list to use the vanilla single layer linear model
        activation_type (str): the activation class name in :obj:`torch.nn`. Set None to
            disable activation and the model is pure linear. Default: None
        activation_conf (dict): the arguments for initializing the activation class.
            Default: empty dict
        pooling_type (str): the pooling class name in :obj:`s3prl.nn.pooling`. Default: MeanPooling
        pooling_conf (dict): the arguments for initializing the pooling class.
            Default: empty dict
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = None,
        activation_type: str = None,
        activation_conf: dict = None,
    ):
        super().__init__()
        self._indim = input_size
        self._outdim = output_size
        hidden_sizes = hidden_sizes or [9,9]
        hidden_sizes.append(output_size)

        latest_size = input_size
        hidden_layers = []
        if len(hidden_sizes) > 0:
            for size in hidden_sizes:
                hidden_layers.append(nn.Linear(latest_size, size))
                if activation_type is not None:
                    hidden_layers.append(
                        getattr(nn, activation_type)(**(activation_conf or {}))
                    )
                latest_size = size

        self.hidden_layers = nn.Sequential(*hidden_layers)

    @property
    def input_size(self) -> int:
        return self._indim

    @property
    def output_size(self) -> int:
        return self._outdim

    def forward(self, x, x_len, mfcc_feat, mfcc_feat_len, handcrafted_features):
        """
        Args:
            x (torch.FloatTensor): (batch_size, seq_len, input_size)
            x_len (torch.LongTensor): (batch_size, )
        Returns:
            torch.FloatTensor
            (batch_size, output_size)
        """
        y = self.hidden_layers(handcrafted_features)
        return y