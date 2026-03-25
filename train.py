import json
import os
import time

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from config import (
    CHECKPOINT_DIR, DEVICE,
    EPOCHS, LR, GRAD_CLIP, ALPHA_C, PATIENCE,
    PAD_IDX, START_IDX, END_IDX,
)
from data import get_loaders
from model import EncoderCNN, DecoderLSTM


def pack(tensor, decode_lengths):
    # flatten predictions/targets, keeping only non-padded positions
    return torch.cat([tensor[i, :l] for i, l in enumerate(decode_lengths)], dim=0)


def train_epoch(encoder, decoder, loader, criterion, optimizer):
    encoder.eval()   # encoder is frozen; switch off batch norm / dropout
    decoder.train()

    total_loss  = 0.0
    total_words = 0

    for images, captions, lengths, _ in loader:
        images   = images.to(DEVICE)
        captions = captions.to(DEVICE)
        lengths  = lengths.to(DEVICE)

        encoder_out = encoder(images)
        predictions, caps_sorted, decode_lengths, alphas, _ = decoder(encoder_out, captions, lengths)

        # targets = everything after <start>
        targets = caps_sorted[:, 1:]

        loss = criterion(pack(predictions, decode_lengths), pack(targets, decode_lengths))

        # doubly-stochastic attention regularizer: encourage each pixel to be
        # attended to roughly once across the caption
        loss += ALPHA_C * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(decoder.parameters(), GRAD_CLIP)
        optimizer.step()

        n = pack(targets, decode_lengths).size(0)
        total_loss  += loss.item() * n
        total_words += n

    return total_loss / total_words


def evaluate_bleu(encoder, decoder, loader):
    encoder.eval()
    decoder.eval()

    dataset  = loader.dataset
    smoother = SmoothingFunction().method1

    # build image -> all reference token lists (strip special tokens)
    img_to_refs = {}
    for img_name, encoded in dataset.samples:
        ref = [w for w in encoded if w not in (START_IDX, END_IDX, PAD_IDX)]
        img_to_refs.setdefault(img_name, []).append(ref)

    references = []
    hypotheses = []
    seen       = set()

    with torch.no_grad():
        for images, _, _, img_names in loader:
            images      = images.to(DEVICE)
            encoder_out = encoder(images)

            for i, img_name in enumerate(img_names):
                if img_name in seen:
                    continue
                seen.add(img_name)

                caption, _ = decoder.generate(encoder_out[i].unsqueeze(0))
                hyp  = [w for w in caption if w not in (START_IDX, END_IDX, PAD_IDX)]
                refs = img_to_refs.get(img_name, [[]])

                hypotheses.append(hyp)
                references.append(refs)

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0),        smoothing_function=smoother)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0),    smoothing_function=smoother)
    bleu3 = corpus_bleu(references, hypotheses, weights=(1/3, 1/3, 1/3, 0),  smoothing_function=smoother)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25,)*4,            smoothing_function=smoother)

    return bleu1, bleu2, bleu3, bleu4


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Loading data...")
    train_loader, val_loader, _, vocab = get_loaders()

    encoder = EncoderCNN().to(DEVICE)
    decoder = DecoderLSTM(len(vocab)).to(DEVICE)

    optimizer = Adam(decoder.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX).to(DEVICE)

    best_bleu4 = 0.0
    no_improve = 0
    history    = []
    best_path  = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    print(f"Training on {DEVICE} for up to {EPOCHS} epochs\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss = train_epoch(encoder, decoder, train_loader, criterion, optimizer)
        b1, b2, b3, b4 = evaluate_bleu(encoder, decoder, val_loader)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d} | loss {loss:.4f} | "
              f"BLEU-1 {b1:.4f} | BLEU-2 {b2:.4f} | "
              f"BLEU-3 {b3:.4f} | BLEU-4 {b4:.4f} | {elapsed:.0f}s")

        history.append(dict(epoch=epoch, loss=loss, bleu1=b1, bleu2=b2, bleu3=b3, bleu4=b4))

        if b4 > best_bleu4:
            best_bleu4 = b4
            no_improve = 0
            torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(), "bleu4": b4, "vocab": vocab}, best_path)
            print(f"  -> saved best model (BLEU-4 = {b4:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping.")
                break

    with open(os.path.join(CHECKPOINT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best BLEU-4: {best_bleu4:.4f}")


if __name__ == "__main__":
    main()
