import torch
import torch.nn as nn
import torchvision.models as models

from config import (
    ENCODER_DIM, ATTENTION_DIM, EMBED_DIM, DECODER_DIM, DROPOUT,
    MAX_CAPTION_LENGTH, PAD_IDX, START_IDX, END_IDX,
)


class EncoderCNN(nn.Module):
    # VGG16 with the last max-pool removed → 14x14x512 feature maps
    # Output: (batch, 196, 512)
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # features[:30] keeps everything up to (not including) pool5
        self.features = nn.Sequential(*list(vgg.features.children())[:30])
        self.pool = nn.AdaptiveAvgPool2d((14, 14))
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, images):
        x = self.features(images)   # (B, 512, h, w)
        x = self.pool(x)            # (B, 512, 14, 14)
        x = x.permute(0, 2, 3, 1)  # (B, 14, 14, 512)
        x = x.reshape(x.size(0), -1, x.size(-1))  # (B, 196, 512)
        return x


class Attention(nn.Module):
    # Soft attention: e_ti = w^T tanh(W_a * a_i + U_a * h_{t-1})
    #                 alpha = softmax(e)
    #                 context = sum_i alpha_i * a_i
    def __init__(self):
        super().__init__()
        self.W_a = nn.Linear(ENCODER_DIM,   ATTENTION_DIM, bias=False)
        self.U_a = nn.Linear(DECODER_DIM,   ATTENTION_DIM, bias=False)
        self.w   = nn.Linear(ATTENTION_DIM, 1,             bias=False)

    def forward(self, encoder_out, h):
        # encoder_out: (B, L, encoder_dim)
        # h:           (B, decoder_dim)
        att = self.W_a(encoder_out) + self.U_a(h).unsqueeze(1)  # (B, L, attn_dim)
        energy = self.w(torch.tanh(att)).squeeze(2)              # (B, L)
        alpha  = torch.softmax(energy, dim=1)                    # (B, L)
        context = (encoder_out * alpha.unsqueeze(2)).sum(1)      # (B, encoder_dim)
        return context, alpha


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD_IDX)
        self.attention = Attention()
        self.lstm      = nn.LSTMCell(EMBED_DIM + ENCODER_DIM, DECODER_DIM)
        self.fc        = nn.Linear(DECODER_DIM, vocab_size)
        self.dropout   = nn.Dropout(DROPOUT)

        # initialize h and c from the mean of image features
        self.init_h = nn.Linear(ENCODER_DIM, DECODER_DIM)
        self.init_c = nn.Linear(ENCODER_DIM, DECODER_DIM)

    def init_hidden(self, encoder_out):
        mean = encoder_out.mean(dim=1)
        h = torch.tanh(self.init_h(mean))
        c = torch.tanh(self.init_c(mean))
        return h, c

    def forward(self, encoder_out, captions, lengths):
        # sort batch by descending length so we can stop early each step
        lengths, sort_idx = lengths.sort(descending=True)
        encoder_out = encoder_out[sort_idx]
        captions    = captions[sort_idx]

        embeddings = self.embedding(captions)  # (B, T, embed_dim)
        h, c = self.init_hidden(encoder_out)

        # decode_lengths = how many steps each sample needs
        # (lengths - 1 because the last input is the word before <end>)
        decode_lengths = (lengths - 1).tolist()
        max_t = int(max(decode_lengths))

        B = encoder_out.size(0)
        L = encoder_out.size(1)
        predictions = torch.zeros(B, max_t, self.vocab_size).to(encoder_out.device)
        alphas      = torch.zeros(B, max_t, L).to(encoder_out.device)

        for t in range(max_t):
            # only process samples that still have tokens at step t
            Bt = sum(dl > t for dl in decode_lengths)

            context, alpha = self.attention(encoder_out[:Bt], h[:Bt])
            h_new, c_new   = self.lstm(
                torch.cat([embeddings[:Bt, t], context], dim=1),
                (h[:Bt], c[:Bt])
            )

            predictions[:Bt, t] = self.fc(self.dropout(h_new))
            alphas[:Bt, t]      = alpha

            # update h and c in place (clone to avoid in-place autograd issues)
            h = h.clone(); c = c.clone()
            h[:Bt] = h_new
            c[:Bt] = c_new

        return predictions, captions, decode_lengths, alphas, sort_idx

    @torch.no_grad()
    def generate(self, encoder_out, max_len=MAX_CAPTION_LENGTH):
        # greedy decoding for a single image
        device = encoder_out.device
        h, c   = self.init_hidden(encoder_out)

        word    = torch.tensor([START_IDX], device=device)
        caption = [START_IDX]
        alphas  = []

        for _ in range(max_len):
            emb              = self.embedding(word)           # (1, embed_dim)
            context, alpha   = self.attention(encoder_out, h)
            h, c             = self.lstm(torch.cat([emb, context], dim=1), (h, c))
            word             = self.fc(h).argmax(1)           # (1,)

            caption.append(word.item())
            alphas.append(alpha.squeeze(0).cpu())

            if word.item() == END_IDX:
                break

        return caption, alphas
