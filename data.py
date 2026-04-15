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
    MAX_CAPTION_LENGTH,
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
            freq.update(cap.lower().strip().split())
        vocab = freq.most_common(3000)

        for word, i in vocab:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, caption):
        tokens = caption.lower().strip().split()
        indices = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        return [START_IDX] + indices + [END_IDX]

    def decode(self, indices):
        words = []
        for i in indices:
            if i == END_IDX:
                break
            if i not in (PAD_IDX, START_IDX):
                words.append(self.idx2word.get(i, UNK_TOKEN))
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


def load_captions(path=CAPTIONS_FILE):
    df = pd.read_csv(path)
    img_to_caps = {}
    for i, row in df.iterrows():
        img = row["image"]
        cap = row["caption"]
        if img not in img_to_caps:
            img_to_caps[img] = []
        img_to_caps[img].append(cap)
    return img_to_caps


def split_data(image_names):
    names = sorted(image_names)
    np.random.seed(50)
    np.random.shuffle(names)
    train = names[:6000]
    val = names[6000:7000]
    test = names[7000:8000]
    return train, val, test


class Flickr8kDataset(Dataset):
    def __init__(self, image_names, img_to_caps, vocab, image_dir=IMAGE_DIR, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.samples = [
            (img, vocab.encode(cap))
            for img in image_names
            for cap in img_to_caps[img]
        ]
        self.img_to_refs = {}
        for img, encoded in self.samples:
            if img not in self.img_to_refs:
                self.img_to_refs[img] = []
            self.img_to_refs[img].append(encoded)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, cap = self.samples[idx]

        image = Image.open(f"{self.image_dir}/{img}").convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(cap), len(cap), img

def collate_fn(batch):
    images = []
    captions = []
    lengths = []
    img_names = []
    for item in batch:
        image, caption, length, img_name = item
        images.append(image)
        captions.append(caption)
        lengths.append(length)
        img_names.append(img_name)
    images = torch.stack(images)
    max_len = max(lengths)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap

    return images, padded, torch.tensor(lengths), img_names


def get_transforms():
    # normalize values from imagenet 
    values = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    return values


def get_loaders(captions_file=CAPTIONS_FILE, image_dir=IMAGE_DIR):
    img_to_caps = load_captions(captions_file)
    train_imgs, val_imgs, test_imgs = split_data(list(img_to_caps.keys()))
    train_caps = []
    for img in train_imgs:
        for cap in img_to_caps[img]:
            train_caps.append(cap)
    vocab = Vocabulary()
    vocab.build(train_caps)
    transform = get_transforms()

    train_ds = Flickr8kDataset(train_imgs, img_to_caps, vocab, image_dir, transform)
    val_ds = Flickr8kDataset(val_imgs, img_to_caps, vocab, image_dir, transform)
    test_ds = Flickr8kDataset(test_imgs, img_to_caps, vocab, image_dir, transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, vocab