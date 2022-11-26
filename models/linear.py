# - ref: https://github.com/s3prl/s3prl/blob/a80b403d7c5f3005beef8cc1086c2fe8dea25cc9/s3prl/nn/linear.py
"""
Common linear models
Authors:
  * Leo 2022
"""

from .common import UtteranceLevel

class MeanPoolingLinear(UtteranceLevel):
    """
    The utterance-level linear probing model used in SUPERB Benchmark
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
    ):
        super().__init__(input_size, output_size, hidden_sizes=[hidden_size])