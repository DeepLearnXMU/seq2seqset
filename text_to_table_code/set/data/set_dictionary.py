# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import Counter
from multiprocessing import Pool

import torch
from fairseq import utils
from fairseq.data import data_utils
from fairseq.file_chunker_utils import Chunker, find_offsets
from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line
from fairseq.data.dictionary import Dictionary

import re
row_header_re = re.compile(r"^91 .*? 930")

class SetDictionary(Dictionary):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        *,  # begin keyword-only arguments
        bos="<s>",
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        extra_special_symbols=None,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def null(self):
        # use the last index as null
        assert len(self) == 51200
        return self.__len__() - 1

    def string(
        self,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
        include_eos=False,
        separator=" ",
        row_score=None,
    ):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if include_eos:
            raise NotImplementedError("include_eos is not supported for SetDictionary")
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(
                self.string(
                    t,
                    bpe_symbol,
                    escape_unk,
                    extra_symbols_to_ignore,
                    include_eos=include_eos,
                )
                for t in tensor
            )

        extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])
        if not include_eos:
            extra_symbols_to_ignore.add(self.eos())

        def token_string(i):
            if i == self.unk():
                if unk_string is not None:
                    return unk_string
                else:
                    return self.unk_string(escape_unk)
            else:
                return self[i]

        if hasattr(self, "bos_index"):
            extra_symbols_to_ignore.add(self.bos())
        sent_list = []
        last = 0
        idx = 0
        row_score_idx = []
        for i, token in enumerate(tensor):
            if token == self.eos() and i != last:
                # if i - last > 2:
                sent_list.append(separator.join(token_string(t) for t in tensor[last:i]))
                last = i + 1
                row_score_idx.append(idx)
            elif token == self.null() or token == self.eos():
                last = i + 1
            if token == self.eos():
                idx += 1
        # sent_list.append(sent_i)
        # filterd_sent = []
        # for i in sent_list:
        #     if re.match(r'^91 ((?!930).)+ 930 ((?!930).)+ 930$', i) is not None:
        #         filterd_sent.append(i)
        # sent = ' 198 '.join(filterd_sent)
        
        if row_score is not None: # 重复row header取score最大的
            row_header_freq = {}
            new_sent_list = []
            for idx, sent in enumerate(sent_list):
                row_header = row_header_re.match(sent)
                if row_header is None:
                    print(sent)
                    continue
                row_header = row_header[0]
                if row_header not in row_header_freq:
                    row_header_freq[row_header] = []
                row_header_freq[row_header].append((idx, row_score[row_score_idx[idx]]))
            for row_header in row_header_freq:
                if len(row_header_freq[row_header]) > 1:
                    max_idx = max(row_header_freq[row_header], key=lambda x: x[1])[0]
                    new_sent_list.append(sent_list[max_idx])
                else:
                    new_sent_list.append(sent_list[row_header_freq[row_header][0][0]])
            sent_list = new_sent_list
        else:
            sent_list = set(sent_list)

        sent = ' 198 '.join(sent_list)
        return data_utils.post_process(sent, bpe_symbol)