# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from omegaconf import II

import numpy as np
from fairseq import metrics, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
    TokenBlockDataset,
    DenoisingDataset
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import TranslationTask, TranslationConfig
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.data.encoders.utils import get_whole_word_mask

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)

SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])
MASK_LENGTH_CHOICES = ChoiceEnum(["subword", "word", "span-poisson"])

@dataclass
class ParadiseConfig(TranslationConfig):
    """ Inherit all the fields from TranslationConfig, add in arguments from denoising
    """
    bpe: Optional[str] = field(
        default=None,
        metadata={"help": "TODO"},
    )

    tokens_per_sample: int = field(
        default=512,
        metadata={
            "help": "max number of total tokens over all segments "
                    "per sample for dataset"
        },
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="eos",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
                    'tokens. If set to "complete", splits samples only at the end '
                    "of sentence, but may include multiple sentences per sample. "
                    '"complete_doc" is similar but respects doc boundaries. '
                    'If set to "eos", includes only one sentence per sample.'
        },
    )
    replace_length: int = field(
        default=0,
        metadata={"help": "TODO, should only allow -1, 0 and 1"},
    )
    mask: float = field(
        default=0.0,
        metadata={"help": "fraction of words/subwords that will be masked"},
    )
    mask_random: float = field(
        default=0.0,
        metadata={"help": "instead of using [MASK], use random token this often"},
    )
    insert: float = field(
        default=0.0,
        metadata={"help": "insert this percentage of additional random tokens"},
    )
    permute: float = field(
        default=0.0,
        metadata={"help": "take this proportion of subwords and permute them"},
    )
    rotate: float = field(
        default=0.5,
        metadata={"help": "rotate this proportion of inputs"},
    )
    poisson_lambda: float = field(
        default=3.0,
        metadata={"help": "randomly shuffle sentences for this proportion of inputs"},
    )
    shuffle_instance: float = field(
        default=0.0,
        metadata={"help": "shuffle this proportion of sentences in all inputs"},
    )
    mask_length: MASK_LENGTH_CHOICES = field(
        default="subword",
        metadata={"help": "mask length to choose"},
    )
    permute_sentences: int = field(
        default=-1,
        metadata={
            "help": "when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)"
        },
    )
    seed: int = II("common.seed")
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
                    'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    add_lang_token: bool = field(  # Note: set to True
        default=True,
        metadata={"help": "whether to add lang tokens in dictionary"},
    )


@register_task("paradise", dataclass=ParadiseConfig)
class ParadiseTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: ParadiseConfig

    def __init__(self, cfg: ParadiseConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.mask_idx = self.src_dict.add_symbol("<mask>")
        self.mask_idx = self.tgt_dict.add_symbol("<mask>")

    @classmethod
    def setup_task(cls, cfg: ParadiseConfig, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))


        #####
        if cfg.add_lang_token:
            langs_load = "ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,af_ZA,az_AZ,bn_IN,fa_IR,he_IL,hr_HR,id_ID,ka_GE,km_KH,mk_MK,ml_IN,mn_MN,mr_IN,pl_PL,ps_AF,pt_XX,sv_SE,sw_KE,ta_IN,te_IN,th_TH,tl_XX,uk_UA,ur_PK,xh_ZA,gl_ES,sl_SI"
            langs_to_add = langs_load.split(",")
            for lang in langs_to_add:
                src_dict.add_symbol("[{}]".format(lang))
                tgt_dict.add_symbol("[{}]".format(lang))

        logger.info("dictionary: {} types".format(len(src_dict)))
        if not hasattr(cfg, "shuffle_instance"):
            cfg.shuffle_instance = False

        return cls(cfg, src_dict, tgt_dict)


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = self._load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
        )

    def _load_langpair_dataset(self,
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=True, # Switched this to True
        load_alignments=False,
        truncate_source=False,
        append_source_id=True, # Switched this to True
        num_buckets=0,
        shuffle=True,
        pad_to_multiple=1,
        prepend_bos_src=None
    ):
        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
            return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

        src_datasets = []
        tgt_datasets = []

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else "")

            # infer langcode
            if split_exists(split_k, src, tgt, src, data_path):
                prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
            elif split_exists(split_k, tgt, src, src, data_path):
                prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError(
                        "Dataset not found: {} ({})".format(split, data_path)
                    )

            src_dataset = data_utils.load_indexed_dataset(
                prefix + src, src_dict, dataset_impl
            )
            if truncate_source:
                src_dataset = AppendTokenDataset(
                    TruncateDataset(
                        StripTokenDataset(src_dataset, src_dict.eos()),
                        max_source_positions - 1,
                    ),
                    src_dict.eos(),
                )
            src_datasets.append(src_dataset)

            tgt_dataset = data_utils.load_indexed_dataset(
                prefix + tgt, tgt_dict, dataset_impl
            )
            if tgt_dataset is not None:
                tgt_datasets.append(tgt_dataset)

            logger.info(
                "{} {} {}-{} {} examples".format(
                    data_path, split_k, src, tgt, len(src_datasets[-1])
                )
            )

            if not combine:
                break

        assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

        if len(src_datasets) == 1:
            src_dataset = src_datasets[0]
            tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            if len(tgt_datasets) > 0:
                tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
            else:
                tgt_dataset = None

        ########################################################################################
        # tokenizing - preprocessing for denoising
        src_dataset = StripTokenDataset(src_dataset, src_dict.eos())
        src_dataset = maybe_shorten_dataset(
            src_dataset,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.cfg.tokens_per_sample,
            self.cfg.seed,
        )

        src_dataset = TokenBlockDataset(
            src_dataset,
            src_dataset.sizes,
            self.cfg.tokens_per_sample - 2,
            # one less for <s> and one for </s>
            pad=src_dict.pad(),
            eos=src_dict.eos(),
            break_mode='eos', # TODO: 'eos'/self.cfg.sample_break_mode - this is needed to split based on sentences
            document_sep_len=0,
        )
        logger.info("loaded {} blocks from src: {}".format(len(src_dataset), data_path))

        ########################################################################################

        if prepend_bos:
            assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
            src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
            if tgt_dataset is not None:
                tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
        elif prepend_bos_src is not None:
            logger.info(f"prepending src bos: {prepend_bos_src}")
            src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

        eos = None
        if append_source_id:
            src_dataset = AppendTokenDataset(
                src_dataset, src_dict.index("[{}]".format(src))
            )
            if tgt_dataset is not None:
                tgt_dataset = AppendTokenDataset(
                    tgt_dataset, tgt_dict.index("[{}]".format(tgt))
                )
            eos = tgt_dict.index("[{}]".format(tgt))

        align_dataset = None
        if load_alignments:
            align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
            if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
                align_dataset = data_utils.load_indexed_dataset(
                    align_path, None, dataset_impl
                )

        tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

        ########################################################################################
        # Perform the denoising here
        mask_whole_words = (
            get_whole_word_mask(self.cfg.bpe, src_dict)
            if self.cfg.mask_length != "subword"
            else None
        )

        src_dataset = DenoisingDataset(
            src_dataset,
            src_dataset.sizes,
            src_dict,
            self.mask_idx,
            mask_whole_words,
            shuffle=self.cfg.shuffle_instance,
            seed=self.cfg.seed,
            mask=self.cfg.mask,
            mask_random=self.cfg.mask_random,
            insert=self.cfg.insert,
            rotate=self.cfg.rotate,
            permute_sentences=self.cfg.permute_sentences,
            bpe=self.cfg.bpe,
            replace_length=self.cfg.replace_length,
            mask_length=self.cfg.mask_length,
            poisson_lambda=self.cfg.poisson_lambda,
            eos=None
            if not self.cfg.add_lang_token
            else self.src_dict.index("[{}]".format(src)),
        )

        logger.info(
            "Split: {0}, Loaded {1} samples of denoising_dataset".format(
                split,
                len(src_dataset),
            )
        )
        ########################################################################################

        return LanguagePairDataset(
            src_dataset,
            src_dataset.sizes,
            src_dict,
            tgt_dataset,
            tgt_dataset_sizes,
            tgt_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset,
            eos=eos,
            num_buckets=num_buckets,
            shuffle=shuffle,
            pad_to_multiple=pad_to_multiple,
        )