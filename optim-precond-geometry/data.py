import os
import gc
import torch
import time
from datasets import load_dataset
from tokenizers import Tokenizer as HFTokenizer, models, trainers, pre_tokenizers, decoders
from tqdm.auto import tqdm

def prepare_data(dataset_name='dclm-edu', num_train_docs=50000, num_val_docs=2000, vocab_size=8192):
    total_docs = num_train_docs + num_val_docs
    all_texts = []

    if dataset_name == 'dclm-edu':
        files = {
            "train": [
                "hf://datasets/HuggingFaceTB/dclm-edu/data/000_00000.parquet",
                "hf://datasets/HuggingFaceTB/dclm-edu/data/000_00001.parquet",
            ]
        }
        ds = load_dataset(
            "parquet",
            data_files=files,
            split="train",
            streaming=False,
            columns=["text", "edu_int_score"],
            filters=[("edu_int_score", ">=", 3)]
        )
        for i, example in enumerate(tqdm(ds, total=total_docs, desc='Downloading')):
            if i >= total_docs:
                break
            all_texts.append(example['text'])

    elif dataset_name == 'pleias-synth':
        ds = load_dataset('PleIAs/SYNTH', split='train', streaming=True)
        for i, example in enumerate(tqdm(ds, total=total_docs, desc='Downloading')):
            if i >= total_docs:
                break
            parts = []
            if example.get('query'):
                parts.append('Question: ' + example['query'])
            if example.get('synthetic_reasoning'):
                parts.append('Reasoning: ' + example['synthetic_reasoning'])
            if example.get('synthetic_answer'):
                parts.append('Answer: ' + example['synthetic_answer'])
            all_texts.append('\n\n'.join(parts))
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    train_texts = all_texts[:num_train_docs]
    val_texts = all_texts[num_train_docs:]
    
    tok_model = HFTokenizer(models.BPE())
    tok_model.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok_model.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=['<|bos|>'],
        min_frequency=2,
        show_progress=True,
    )
    tok_model.train_from_iterator(train_texts, trainer=trainer)

    BOS_TOKEN_ID = tok_model.token_to_id('<|bos|>')

    def tokenize_texts(texts, desc='Tokenizing'):
        all_ids = []
        batch_size = 1000
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch = texts[i:i + batch_size]
            encoded = tok_model.encode_batch(batch)
            for enc in encoded:
                all_ids.append(BOS_TOKEN_ID)
                all_ids.extend(enc.ids)
        return torch.tensor(all_ids, dtype=torch.long)

    train_data = tokenize_texts(train_texts, desc='Train')
    val_data = tokenize_texts(val_texts, desc='Val')

    del all_texts, train_texts, val_texts
    gc.collect()

    return train_data, val_data, tok_model, BOS_TOKEN_ID


def make_dataloader(data_tensor, batch_size, seq_len, device):
    """
    Infinite dataloader that yields random chunks from a flat token tensor.
    """
    n = len(data_tensor) - seq_len - 1
    assert n > 0, f'Data too short ({len(data_tensor)} tokens) for seq_len={seq_len}'
    while True:
        ix = torch.randint(0, n, (batch_size,))
        x = torch.stack([data_tensor[i:i + seq_len] for i in ix]).to(device)
        y = torch.stack([data_tensor[i + 1:i + seq_len + 1] for i in ix]).to(device)
        yield x, y
