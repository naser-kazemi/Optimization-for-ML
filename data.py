import os
import gc
import torch
import time

def prepare_data(dataset_name='dclm-edu', num_train_docs=50000, num_val_docs=2000,
                 vocab_size=8192, cache_dir=None):
    """Build (or load) the tokenizer + tokenized train/val tensors.

    If `cache_dir` is set, the tokenizer (`tokenizer.json`) and token tensors
    (`train_data.pt`, `val_data.pt`) are persisted on first call and reloaded on
    later calls. This is required for the connectivity experiment: every endpoint
    training and the connecting run must share ONE identical tokenizer + data so
    that the loss they are evaluated under is the same function. By default
    (`cache_dir=None`) behavior is unchanged — a fresh tokenizer per call.
    """
    # Heavy data-stack imports are lazy so that lightweight helpers
    # (make_dataloader, make_fixed_eval_batches) can be imported without the
    # HuggingFace datasets/tokenizers packages installed.
    from datasets import load_dataset
    from tokenizers import Tokenizer as HFTokenizer, models, trainers, pre_tokenizers, decoders
    from tqdm.auto import tqdm

    if cache_dir is not None:
        tok_path = os.path.join(cache_dir, 'tokenizer.json')
        train_path = os.path.join(cache_dir, 'train_data.pt')
        val_path = os.path.join(cache_dir, 'val_data.pt')
        if all(os.path.exists(p) for p in (tok_path, train_path, val_path)):
            print(f"Loading cached data + tokenizer from '{cache_dir}'")
            tok_model = HFTokenizer.from_file(tok_path)
            train_data = torch.load(train_path)
            val_data = torch.load(val_path)
            bos_id = tok_model.token_to_id('<|bos|>')
            return train_data, val_data, tok_model, bos_id

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

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        tok_model.save(os.path.join(cache_dir, 'tokenizer.json'))
        torch.save(train_data, os.path.join(cache_dir, 'train_data.pt'))
        torch.save(val_data, os.path.join(cache_dir, 'val_data.pt'))
        print(f"Saved data + tokenizer to '{cache_dir}'")

    return train_data, val_data, tok_model, BOS_TOKEN_ID


def make_fixed_eval_batches(data_tensor, batch_size, seq_len, n_batches, device, seed=1234):
    """Sample a fixed, deterministic set of (x, y) batches.

    Unlike `make_dataloader` (an infinite random generator), this returns the
    SAME batches every call for a given seed, giving NEB a stable, reproducible
    loss landscape to optimize and measure on.
    """
    n = len(data_tensor) - seq_len - 1
    assert n > 0, f'Data too short ({len(data_tensor)} tokens) for seq_len={seq_len}'
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        ix = torch.randint(0, n, (batch_size,), generator=g)
        x = torch.stack([data_tensor[i:i + seq_len] for i in ix]).to(device)
        y = torch.stack([data_tensor[i + 1:i + seq_len + 1] for i in ix]).to(device)
        batches.append((x, y))
    return batches


def make_dataloader(data_tensor, batch_size, seq_len, device, seed=None):
    """
    Infinite dataloader that yields random chunks from a flat token tensor.
    With `seed` set the batch order is reproducible (private generator).
    """
    n = len(data_tensor) - seq_len - 1
    assert n > 0, f'Data too short ({len(data_tensor)} tokens) for seq_len={seq_len}'
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    while True:
        ix = torch.randint(0, n, (batch_size,), generator=g)
        x = torch.stack([data_tensor[i:i + seq_len] for i in ix]).to(device)
        y = torch.stack([data_tensor[i + 1:i + seq_len + 1] for i in ix]).to(device)
        yield x, y
