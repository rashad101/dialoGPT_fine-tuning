import pickle

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
import os, logging
logger = logging.getLogger(__name__)

class ConversationDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, df, block_size=512):

        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        print(tokenizer.model_max_length, tokenizer.max_len_single_sentence)
        print(block_size)
        directory = args.cache_dir
        cached_features_file = os.path.join(directory, args.model_type + "_cached_lm_"+str(block_size))

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.exmaple = []

            for _, row in df.iterrows():
                conv = self.construct_conv(row, tokenizer)
                self.exmaple.append(conv)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.exmaple, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.exmaple)

    def __getitem__(self, item):
        return torch.tensor(self.exmaple[item], dtype=torch.long)

    @staticmethod
    def construct_conv(row, tokenizer, eos=True):
        flatten = lambda l: [item for sublist in l for item in sublist]
        conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
        conv = flatten(conv)
        return conv

