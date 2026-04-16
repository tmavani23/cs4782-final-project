import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision.transforms as T

from config import CHECKPOINT_DIR, DEVICE, IMAGE_SIZE, START_IDX, END_IDX, PAD_IDX
from model import EncoderCNN, DecoderLSTM


def load_model(checkpoint_path):
    ckpt    = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    vocab   = ckpt["vocab"]
    encoder = EncoderCNN().to(DEVICE)
    decoder = DecoderLSTM(len(vocab)).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, vocab


def preprocess(image_path):
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)


def visualize(image_path, caption, attn_maps, vocab, out_path="attention.png", max_words=20):
    # load image for display
    img = np.array(Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))

    # collect (word, attention) pairs, skipping special tokens
    words_attn = []
    for step, tok in enumerate(caption[1:]):   # skip <start>
        if tok in (START_IDX, END_IDX, PAD_IDX):
            continue
        words_attn.append((vocab.idx2word.get(tok, "?"), attn_maps[step]))
        if len(words_attn) == max_words:
            break

    n     = len(words_attn)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (word, alpha) in zip(axes_flat, words_attn):
        # reshape 196 -> 14x14 and upsample to IMAGE_SIZE x IMAGE_SIZE using PIL
        attn_img = Image.fromarray(alpha.numpy().reshape(14, 14))
        attn_img = attn_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        attn_arr = np.array(attn_img)

        ax.imshow(img)
        ax.imshow(attn_arr, alpha=0.5, cmap="jet")
        ax.set_title(word, fontsize=10)
        ax.axis("off")

    for ax in axes_flat[n:]:
        ax.axis("off")

    caption_str = " ".join(w for w, _ in words_attn)
    fig.suptitle(f'"{caption_str}"', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved attention map to {out_path}")
    print(f"Caption: {caption_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True)
    parser.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "best_model.pth"))
    parser.add_argument("--out",        default="attention.png")
    parser.add_argument("--max_words",  type=int, default=20)
    args = parser.parse_args()

    encoder, decoder, vocab = load_model(args.checkpoint)

    image_tensor = preprocess(args.image)
    with torch.no_grad():
        encoder_out       = encoder(image_tensor)
        caption, attn_maps = decoder.generate(encoder_out)

    visualize(args.image, caption, attn_maps, vocab, args.out, args.max_words)


if __name__ == "__main__":
    main()
