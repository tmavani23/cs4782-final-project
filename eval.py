import argparse
import json
import os

import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from config import CHECKPOINT_DIR, DEVICE, CAPTIONS_FILE, IMAGE_DIR, START_IDX, END_IDX, PAD_IDX
from data import get_loaders
from model import EncoderCNN, DecoderLSTM


def load_model(checkpoint_path):
    ckpt    = torch.load(checkpoint_path, map_location=DEVICE)
    vocab   = ckpt["vocab"]
    encoder = EncoderCNN().to(DEVICE)
    decoder = DecoderLSTM(len(vocab)).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    print(f"Loaded epoch {ckpt['epoch']} checkpoint (val BLEU-4 = {ckpt['bleu4']:.4f})")
    return encoder, decoder, vocab


def run_eval(encoder, decoder, loader):
    dataset = loader.dataset

    # image -> all reference token lists (strip special tokens)
    img_to_refs = {}
    for img_name, encoded in dataset.samples:
        ref = [w for w in encoded if w not in (START_IDX, END_IDX, PAD_IDX)]
        img_to_refs.setdefault(img_name, []).append(ref)

    references = []
    hypotheses = []
    img_names  = []
    seen       = set()

    with torch.no_grad():
        for images, _, _, names in loader:
            images      = images.to(DEVICE)
            encoder_out = encoder(images)

            for i, img_name in enumerate(names):
                if img_name in seen:
                    continue
                seen.add(img_name)

                caption, _ = decoder.generate(encoder_out[i].unsqueeze(0))
                hyp  = [w for w in caption if w not in (START_IDX, END_IDX, PAD_IDX)]
                refs = img_to_refs.get(img_name, [[]])

                hypotheses.append(hyp)
                references.append(refs)
                img_names.append(img_name)

    smoother = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0),       smoothing_function=smoother)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0),   smoothing_function=smoother)
    bleu3 = corpus_bleu(references, hypotheses, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoother)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25,)*4,           smoothing_function=smoother)

    return bleu1, bleu2, bleu3, bleu4, hypotheses, references, img_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "best_model.pth"))
    parser.add_argument("--split",      default="test", choices=["val", "test"])
    parser.add_argument("--samples",    type=int, default=10)
    args = parser.parse_args()

    print("Loading data...")
    train_loader, val_loader, test_loader, _ = get_loaders()
    encoder, decoder, vocab = load_model(args.checkpoint)

    loader = test_loader if args.split == "test" else val_loader
    print(f"Evaluating on {args.split} set...")

    b1, b2, b3, b4, hypotheses, references, img_names = run_eval(encoder, decoder, loader)

    print(f"\nBLEU-1: {b1:.4f}")
    print(f"BLEU-2: {b2:.4f}")
    print(f"BLEU-3: {b3:.4f}")
    print(f"BLEU-4: {b4:.4f}")
    print("(Paper reports: B-1=0.67  B-2=0.45  B-3=0.31  B-4=0.21 for soft-att on Flickr8k)")

    # print some example predictions
    print("\nSample predictions:")
    for idx in range(min(args.samples, len(hypotheses))):
        print(f"\n  Image : {img_names[idx]}")
        print(f"  Pred  : {vocab.decode(hypotheses[idx])}")
        for j, ref in enumerate(references[idx][:2]):
            print(f"  Ref {j+1} : {vocab.decode(ref)}")

    # save results
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    results = dict(split=args.split, bleu1=b1, bleu2=b2, bleu3=b3, bleu4=b4,
                   num_images=len(hypotheses))
    out_path = os.path.join(CHECKPOINT_DIR, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
