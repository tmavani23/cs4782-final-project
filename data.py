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
    IMAGE_DIR, CAPTIONS_FILE,
    PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN,
    PAD_IDX, START_IDX, END_IDX, UNK_IDX,
    BATCH_SIZE,
)

# vocab class
class Vocabulary:
    def __init__(self):
        self.word2idx = {PAD_TOKEN: PAD_IDX, START_TOKEN: START_IDX,
                         END_TOKEN: END_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    # counts all words from training captions and keeps 3000 most commmon ones and adds them to vocab
    def build(self, captions):
        freq = Counter()
        for cap in captions:
            freq.update(tokenize(cap))
        vocab = freq.most_common(10000)

        for word, i in vocab:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    # turns a caption into token IDs and wraps it in start and end tokens
    def encode(self, caption):
        tokens = tokenize(caption)
        indices = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        return ([START_IDX] + indices + [END_IDX])

    # turns IDs back into readable sentences
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

def tokenize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)   # remove punctuation
    tokens = text.split()
    return tokens

# reads csv and makes a dict mapping each image filename to its list of captions
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

# shuffles image names with a fixed seed and seperates image names into 6000 train, 1000 val, 1000 test
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
        self.vocab = vocab
        self.transform = transform

        # each sample is one (image, encoded caption) so if an image has 5 captions then it is in the datatset 5 times
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
    padded = torch.full((len(captions), max_len), PAD_IDX, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap

    return images, padded, torch.tensor(lengths), img_names

def get_transforms():
    train_transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, eval_transform

# loads captions, splits images, builds vocab from only training captions
def get_loaders(captions_file=CAPTIONS_FILE, image_dir=IMAGE_DIR):
    img_to_caps = load_captions(captions_file)
    train_imgs, val_imgs, test_imgs = split_data(list(img_to_caps.keys()))
    
    train_caps = []
    for img in train_imgs:
        for cap in img_to_caps[img]:
            train_caps.append(cap)
    vocab = Vocabulary()
    vocab.build(train_caps)
    train_transform, eval_transform = get_transforms()

    train_ds = Flickr8kDataset(train_imgs, img_to_caps, vocab, image_dir, train_transform)
    val_ds = Flickr8kDataset(val_imgs, img_to_caps, vocab, image_dir, eval_transform)
    test_ds = Flickr8kDataset(test_imgs, img_to_caps, vocab, image_dir, eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, vocab


def test():
    print("Checking data file")
    train_loader, val_loader, test_loader, vocab = get_loaders()
    print("Vocab size:", len(vocab))
    images, captions, lengths, img_names = next(iter(train_loader))
    print("Images shape:", images.shape)
    print("Captions shape:", captions.shape)
    print("Some caption lengths:", lengths[:5])
    print("One example image name:", img_names[0])
    print("One decoded caption:", vocab.decode(captions[0].tolist()))

if __name__ == "__main__":
    test()