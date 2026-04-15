import argparse
import json
import os
import torch
from nltk.translate.bleu_score import corpus_bleu
from config import CHECKPOINT_DIR, DEVICE, START_IDX, END_IDX, PAD_IDX
from data import get_loaders
from model import EncoderCNN, DecoderLSTM

def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    vocab = ckpt["vocab"]
    encoder = EncoderCNN().to(DEVICE)
    decoder = DecoderLSTM(len(vocab)).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, vocab

def run_eval(encoder, decoder, loader):
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
    img_names = []
    seen = set()

    with torch.no_grad():
        for images, _, _, names in loader:
            images = images.to(DEVICE)
            encoder_out = encoder(images)

            for i, img_name in enumerate(names):
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
                img_names.append(img_name)

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(1/3, 1/3, 1/3, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu1, bleu2, bleu3, bleu4, hypotheses, references, img_names


def main():
    train_loader, val_loader, test_loader, _ = get_loaders()
    encoder, decoder, vocab = load_model(os.path.join(CHECKPOINT_DIR, "best_model.pth"))

    bleu1, bleu2, bleu3, bleu4, hypotheses, references, img_names = run_eval(
        encoder, decoder, test_loader
    )

    print("BLEU-1:", bleu1)
    print("BLEU-2:", bleu2)
    print("BLEU-3:", bleu3)
    print("BLEU-4:", bleu4)

    print("\nSample predictions:")
    for i in range(min(10, len(hypotheses))):
        print("\nImage:", img_names[i])
        print("Pred:", vocab.decode(hypotheses[i]))
        for j, ref in enumerate(references[i][:2]):
            print("Ref", j + 1, ":", vocab.decode(ref))

if __name__ == "__main__":
    main()
