# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from doctest import Example
import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils


logger = logging.getLogger(__name__)

NEWLINE = 50118
EOS = 2
NULL = 51199
BOS_0 = 50361
SEP = 1721

MAP = {10704: 0, 3652: 1, 3655: 2, 19458: 3, 3664: 4, 4121: 5, 15729: 6,}

def collate(
    samples,
    pad_idx,
    eos_idx,
    max_num,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )
    
    def merge_set(max_num, move_eos_to_beginning=False, return_segment=False):
        """Convert a list of 1d tensors into a padded 3d tensor."""
        data = [s["target"] for s in samples]
        header_data = []
        non_header_data = []
        header_seg = []
        non_header_seg = []
        non_header_max_len = 0
        assert data[0][-1] == EOS
        if return_segment:
            for d in data:
                header = True
                last = 0
                seg_idx = 0
                set_d = []
                set_s = []
                s = []
                for idx, v in enumerate(d):
                    s.append(seg_idx)
                    if v == NEWLINE or v == EOS:
                        if header:
                            header_data.append(d[last:idx+1])
                            header_seg.append(torch.tensor(s))
                            header = False
                        else:
                            set_d.append(d[last:idx+1])
                            set_s.append(torch.tensor(s))
                        last = idx + 1
                        s = []
                        seg_idx = 0
                    elif v == SEP:
                        seg_idx += 1
                assert not header
                assert last == len(d)
                non_header_max_len = max(non_header_max_len, max(len(i) for i in set_d))
                non_header_data.append(set_d)
                non_header_seg.append(set_s)
        else:
            for d in data:
                header = True
                last = 0
                set_d = []
                for idx, v in enumerate(d):
                    if v == NEWLINE or v == EOS:
                        if header:
                            header_data.append(d[last:idx+1])
                            header = False
                        else:
                            set_d.append(d[last:idx+1])
                        last = idx + 1
                assert not header
                assert last == len(d)
                non_header_max_len = max(non_header_max_len, max(len(i) for i in set_d))
                non_header_data.append(set_d)
        header_max_len = max(len(i) for i in header_data)
        
        # batch_size, max_row, max_len
        res = data[0][0].new_full((non_header_max_len, ), pad_idx)
        if move_eos_to_beginning:
            res[0] = EOS
            res[1] = NULL
        else:
            res[0] = NULL
            res[1] = EOS
        
        res = res.reshape(1, 1, non_header_max_len).repeat(len(data), max_num-1, 1)
        hed = data[0][0].new_full((len(data), 1, header_max_len), pad_idx)
        if return_segment:
            seg = torch.zeros_like(res)
            hed_seg = torch.zeros_like(hed)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                dst[0] = EOS
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)
                dst[-1] = EOS

        def copy_seg(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)


        if return_segment:
            for i in range(len(non_header_data)):
                for j in range(len(non_header_data[i])):
                    copy_tensor(non_header_data[i][j], res[i][j][: len(non_header_data[i][j])])
                    copy_seg(non_header_seg[i][j], seg[i][j][: len(non_header_seg[i][j])])
                copy_tensor(header_data[i], hed[i][0][: len(header_data[i])])
                copy_seg(header_seg[i], hed_seg[i][0][: len(header_seg[i])])
            return res, seg, hed, hed_seg
        else:
            for i, u in enumerate(non_header_data):
                for j, v in enumerate(u):
                    copy_tensor(v, res[i][j][: len(v)])
                copy_tensor(header_data[i], hed[i][0][: len(header_data[i])])
            return res, hed

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        non_header_target, non_header_segment, header_target, header_segment = merge_set(
            max_num=max_num, return_segment=True
        )
        non_header_target = non_header_target.index_select(0, sort_order)
        non_header_segment = non_header_segment.index_select(0, sort_order)
        non_header_ntokens = non_header_target.ne(pad_idx).sum().item()
        header_target = header_target.index_select(0, sort_order)
        header_segment = header_segment.index_select(0, sort_order)
        header_ntokens = header_target.ne(pad_idx).sum().item()
        target = (header_target, non_header_target)
        target_segment = (header_segment, non_header_segment)
        ntokens = (header_ntokens, non_header_ntokens)
        # tgt_lengths = torch.LongTensor(
        #     [s["target"].ne(pad_idx).long().sum() for s in samples]
        # ).index_select(0, sort_order)
        # tgt_lengths = target.reshape(sort_order.size(0), -1).ne(pad_idx).sum(1).index_select(0, sort_order)

        if samples[0].get("prev_output_tokens", None) is not None:
            raise NotImplementedError
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            non_header_prev_output_tokens, header_prev_output_tokens = merge_set(
                max_num = max_num,
                move_eos_to_beginning=True,
            )
            non_header_prev_output_tokens = non_header_prev_output_tokens.index_select(0, sort_order)
            header_prev_output_tokens = header_prev_output_tokens.index_select(0, sort_order)
            prev_output_tokens = (header_prev_output_tokens, non_header_prev_output_tokens)
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens
        batch["net_input"]["segment_ids"] = target_segment

    return batch


class Seq2Set3LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
        self,
        max_num,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
    ):
        self.max_num = max_num
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes[:, 0]*(max_num-1)+self.tgt_sizes[:, 1])).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info(
                    "bucketing target lengths: {}".format(list(self.tgt.buckets))
                )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
        }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            max_num=self.max_num,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index, 0]*(self.max_num-1)+self.tgt_sizes[index, 1] if self.tgt_sizes is not None else 0,
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index, 0]*(self.max_num-1)+self.tgt_sizes[index, 1] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices][:, 0], kind="mergesort")]
                indices = indices[np.argsort(self.tgt_sizes[indices][:, 1], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        tgt_sizes = self.tgt_sizes[:, 0]*(self.max_num-1)+self.tgt_sizes[:, 1] if self.tgt_sizes is not None else None
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            tgt_sizes,
            indices,
            max_sizes,
        )
