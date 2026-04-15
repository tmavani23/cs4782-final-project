import json
import os
import time

import torch
import torch.nn as nn
from torch.optim import RMSprop
from nltk.translate.bleu_score import corpus_bleu

from config import (
    CHECKPOINT_DIR, DEVICE,
    EPOCHS, LR, ALPHA_C,
    PAD_IDX, START_IDX, END_IDX,
)
from data import get_loaders
from model import EncoderCNN, DecoderLSTM


def pack(tensor, lengths):
    result = []
    for i in range(len(lengths)):
        l = lengths[i]
        result.append(tensor[i, :l])
    return torch.cat(result, dim=0)

def train_epoch(encoder, decoder, loader, criterion, optimizer):
    encoder.eval()  
    decoder.train()
    total_loss  = 0.0
    total_words = 0

    for images, captions, lengths, names in loader:
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)
        lengths = lengths.to(DEVICE)

        encoder_output = encoder(images)
        predictions, caps_sorted, decode_lengths, alphas, sort_idx = decoder(encoder_output, captions, lengths)
        targets = caps_sorted[:, 1:]
        
        loss = criterion(pack(predictions, decode_lengths), pack(targets, decode_lengths))

        loss += ALPHA_C * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = pack(targets, decode_lengths).size(0)
        total_loss  += loss.item() * n
        total_words += n

    return total_loss / total_words


def evaluate_bleu(encoder, decoder, loader):
    encoder.eval()
    decoder.eval()
    dataset  = loader.dataset

    img_to_refs = {}
    for img_name, encoded in dataset.samples:
        clean_caption = []
        for w in encoded:
            if w not in (START_IDX, END_IDX, PAD_IDX):
                clean_caption.append(w)
        if img_name not in img_to_refs:
            img_to_refs[img_name] = []
        img_to_refs[img_name].append(clean_caption)

    references = []
    hypotheses = []
    seen = set()

    with torch.no_grad():
        for images, _, _, img_names in loader:
            images = images.to(DEVICE)
            encoder_out = encoder(images)

            for i in range(len(img_names)):
                img_name = img_names[i]
                if img_name in seen:
                    continue
                seen.add(img_name)
                output, _ = decoder.generate(encoder_out[i].unsqueeze(0))
                clean_pred = []
                for w in output:
                    if w not in (START_IDX, END_IDX, PAD_IDX):
                        clean_pred.append(w)
                refs = img_to_refs[img_name]
                hypotheses.append(clean_pred)
                references.append(refs)
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(1/3, 1/3, 1/3, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu1, bleu2, bleu3, bleu4


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_loader, val_loader, _, vocab = get_loaders()

    encoder = EncoderCNN().to(DEVICE)
    decoder = DecoderLSTM(len(vocab)).to(DEVICE)

    optimizer = RMSprop(decoder.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX).to(DEVICE)

    best_bleu4 = 0
    epochs_no_improve = 0
    history = []

    for epoch in range(EPOCHS):
        loss = train_epoch(encoder, decoder, train_loader, criterion, optimizer)
        bleu1, bleu2, bleu3, bleu4 = evaluate_bleu(encoder, decoder, val_loader)

        print("Epoch:", epoch + 1)
        print("Loss:", loss)
        print("BLEU-1:", bleu1)
        print("BLEU-2:", bleu2)
        print("BLEU-3:", bleu3)
        print("BLEU-4:", bleu4)

        history.append({
            "epoch": epoch + 1,
            "loss": loss,
            "bleu1": bleu1,
            "bleu2": bleu2,
            "bleu3": bleu3,
            "bleu4": bleu4
        })

        if bleu4 > best_bleu4:
            best_bleu4 = bleu4
            epochs_no_improve = 0
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "vocab": vocab
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= 3:
            print("Stopping early")
            break

    with open(os.path.join(CHECKPOINT_DIR, "history.json"), "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    main()
