import json
import os
import time

import torch
import torch.nn as nn
from torch.optim import RMSprop
from nltk.translate.bleu_score import corpus_bleu

from config import (
    CHECKPOINT_DIR, DEVICE,
    EPOCHS, LR, ALPHA_C, PATIENCE,
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


def train_epoch(encoder, decoder, loader, criterion, optimizer, epoch_num):
    encoder.eval()
    decoder.train()

    total_loss = 0.0
    total_words = 0
    num_batches = len(loader)
    start_time = time.time()

    for batch_idx, (images, captions, lengths, names) in enumerate(loader, start=1):
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)
        lengths = lengths.to(DEVICE)
        
        with torch.no_grad():
            encoder_output = encoder(images)

        predictions, caps_sorted, decode_lengths, alphas, sort_idx = decoder(
            encoder_output, captions, lengths
        )
        targets = caps_sorted[:, 1:]

        packed_predictions = pack(predictions, decode_lengths)
        packed_targets = pack(targets, decode_lengths)

        loss = criterion(packed_predictions, packed_targets)
        loss += ALPHA_C * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_words = packed_targets.size(0)
        total_loss += loss.item() * n_words
        total_words += n_words

        if batch_idx % 100 == 0 or batch_idx == 1 or batch_idx == num_batches:
            elapsed = time.time() - start_time
            avg_loss = total_loss / total_words
            print(
                f"[Epoch {epoch_num}] Batch {batch_idx}/{num_batches} "
                f"| batch_loss={loss.item():.4f} | avg_loss={avg_loss:.4f} "
                f"| elapsed={elapsed:.1f}s"
            )

    return total_loss / total_words


def evaluate_bleu(encoder, decoder, loader):
    encoder.eval()
    decoder.eval()
    dataset = loader.dataset

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

    num_batches = len(loader)
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, _, _, img_names) in enumerate(loader, start=1):
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

            if batch_idx % 20 == 0 or batch_idx == 1 or batch_idx == num_batches:
                elapsed = time.time() - start_time
                print(
                    f"[Validation] Batch {batch_idx}/{num_batches} "
                    f"| images_done={len(seen)} | elapsed={elapsed:.1f}s"
                )

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(1/3, 1/3, 1/3, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu1, bleu2, bleu3, bleu4


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_loader, val_loader, _, vocab = get_loaders()

    print(f"Loaded data. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Using device: {DEVICE}")

    encoder = EncoderCNN().to(DEVICE)
    decoder = DecoderLSTM(len(vocab)).to(DEVICE)

    optimizer = RMSprop([{"params": encoder.parameters(), "lr": 1e-5}, {"params": decoder.parameters(), "lr": 5e-5}])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX).to(DEVICE)

    best_bleu4 = 0.0
    epochs_no_improve = 0
    history = []

    total_start = time.time()

    for epoch in range(EPOCHS):
        epoch_num = epoch + 1
        print(f"\n========== Epoch {epoch_num}/{EPOCHS} ==========")

        epoch_start = time.time()
        train_loss = train_epoch(encoder, decoder, train_loader, criterion, optimizer, epoch_num)

        print(f"[Epoch {epoch_num}] Training done. Starting validation...")
        bleu1, bleu2, bleu3, bleu4 = evaluate_bleu(encoder, decoder, val_loader)

        epoch_time = time.time() - epoch_start

        print(f"[Epoch {epoch_num}] Done in {epoch_time:.1f}s")
        print(f"Loss:   {train_loss:.4f}")
        print(f"BLEU-1: {bleu1:.4f}")
        print(f"BLEU-2: {bleu2:.4f}")
        print(f"BLEU-3: {bleu3:.4f}")
        print(f"BLEU-4: {bleu4:.4f}")

        history.append({
            "epoch": epoch_num,
            "loss": train_loss,
            "bleu1": bleu1,
            "bleu2": bleu2,
            "bleu3": bleu3,
            "bleu4": bleu4,
            "epoch_time_sec": epoch_time,
        })

        if bleu4 > best_bleu4:
            best_bleu4 = bleu4
            epochs_no_improve = 0
            save_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "vocab": vocab,
            }, save_path)
            print(f"[Epoch {epoch_num}] New best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"[Epoch {epoch_num}] No improvement. Patience: {epochs_no_improve}/{PATIENCE}")

        if epochs_no_improve >= PATIENCE:
            print("Stopping early.")
            break

    history_path = os.path.join(CHECKPOINT_DIR, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training history to {history_path}")
    print(f"Total runtime: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()