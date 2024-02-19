import re

from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass


@register_tokenizer("set", dataclass=FairseqDataclass)
class SetTokenizer(object):

    def __init__(self, *unused):
        pass

    def encode(self, x: str) -> str:
        return x

    def decode(self, x: str) -> str:
        return x
