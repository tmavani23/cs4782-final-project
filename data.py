import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from config import (
    IMAGE_DIR, CAPTIONS_FILE, IMAGE_SIZE,
    MAX_CAPTION_LENGTH, MIN_WORD_FREQ,
    TRAIN_FRAC, VAL_FRAC,
    PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN,
    PAD_IDX, START_IDX, END_IDX, UNK_IDX,
    BATCH_SIZE,
)


class Vocabulary:
    def __init__(self):
        self.word2idx = {PAD_TOKEN: PAD_IDX, START_TOKEN: START_IDX,
                         END_TOKEN: END_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def build(self, captions):
        freq = Counter()
        for cap in captions:
            freq.update(tokenize(cap))
        for word, cnt in freq.items():
            if cnt >= MIN_WORD_FREQ and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, caption):
        tokens = tokenize(caption)
        return [START_IDX] + [self.word2idx.get(t, UNK_IDX) for t in tokens] + [END_IDX]

    def decode(self, indices):
        words = []
        for i in indices:
            if i in (PAD_IDX, START_IDX):
                continue
            if i == END_IDX:
                break
            words.append(self.idx2word.get(i, UNK_TOKEN))
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


def tokenize(caption):
    caption = caption.lower().strip()
    caption = re.sub(r"[^a-z0-9\s]", "", caption)
    return caption.split()


def load_captions(path=CAPTIONS_FILE):
    # Kaggle Flickr8k captions.txt has columns: image, caption
    df = pd.read_csv(path)
    img_to_caps = {}
    for _, row in df.iterrows():
        img = str(row["image"]).strip()
        cap = str(row["caption"]).strip()
        img_to_caps.setdefault(img, []).append(cap)
    return img_to_caps


def split_data(image_names, seed=42):
    names = sorted(image_names)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    n = len(names)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    return names[:n_train], names[n_train:n_train + n_val], names[n_train + n_val:]


class Flickr8kDataset(Dataset):
    def __init__(self, image_names, img_to_caps, vocab, image_dir=IMAGE_DIR, transform=None):
        self.image_dir = image_dir
        self.vocab     = vocab
        self.transform = transform

        # one entry per (image, caption) pair
        self.samples = []
        for img in image_names:
            for cap in img_to_caps.get(img, []):
                encoded = vocab.encode(cap)
                if len(encoded) <= MAX_CAPTION_LENGTH:
                    self.samples.append((img, encoded))

        # map image name -> all reference encodings (used for BLEU)
        self.img_to_refs = {}
        for img_name, encoded in self.samples:
            self.img_to_refs.setdefault(img_name, []).append(encoded)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, encoded = self.samples[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption = torch.tensor(encoded, dtype=torch.long)
        return image, caption, len(encoded), img_name


def collate_fn(batch):
    images, captions, lengths, img_names = zip(*batch)
    images  = torch.stack(images, 0)
    max_len = max(lengths)
    padded  = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, (cap, l) in enumerate(zip(captions, lengths)):
        padded[i, :l] = cap
    return images, padded, torch.tensor(lengths, dtype=torch.long), img_names


def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_tf = T.Compose([
        T.Resize(256),
        T.RandomCrop(IMAGE_SIZE),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def get_loaders(captions_file=CAPTIONS_FILE, image_dir=IMAGE_DIR):
    img_to_caps = load_captions(captions_file)
    train_imgs, val_imgs, test_imgs = split_data(list(img_to_caps.keys()))

    # build vocab from training captions only
    train_caps = [cap for img in train_imgs for cap in img_to_caps.get(img, [])]
    vocab = Vocabulary()
    vocab.build(train_caps)

    print(f"Vocab size: {len(vocab)}")
    print(f"Train: {len(train_imgs)} images | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    train_tf, eval_tf = get_transforms()
    train_ds = Flickr8kDataset(train_imgs, img_to_caps, vocab, image_dir, train_tf)
    val_ds   = Flickr8kDataset(val_imgs,   img_to_caps, vocab, image_dir, eval_tf)
    test_ds  = Flickr8kDataset(test_imgs,  img_to_caps, vocab, image_dir, eval_tf)

    loader_kw = dict(batch_size=BATCH_SIZE, collate_fn=collate_fn,
                     num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader, vocab
