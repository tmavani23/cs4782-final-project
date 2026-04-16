import torch
import torch.nn as nn
import torchvision.models as models

from data import get_loaders

from config import (
    ENCODER_DIM, ATTENTION_DIM, EMBED_DIM, DECODER_DIM, DROPOUT,
    MAX_CAPTION_LENGTH, PAD_IDX, START_IDX, END_IDX,
)

# takes image and turns it into a set of spatial feature vectors
class EncoderCNN(nn.Module):
    # 14x14x512 feature map flattened into 196 locations of dim 512
    # output: (batch, 196, 512)
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
        x = x.permute(0, 2, 3, 1)  
        x = x.reshape(x.size(0), -1, x.size(-1))  # (B, 196, 512)
        return x

# at each decoding step, we see which spatial features matter the most
class Attention(nn.Module):
    # only soft attention
    def __init__(self):
        super().__init__()
        self.W_a = nn.Linear(ENCODER_DIM, ATTENTION_DIM, bias=False)
        self.U_a = nn.Linear(DECODER_DIM, ATTENTION_DIM, bias=False)
        self.w = nn.Linear(ATTENTION_DIM, 1, bias=False)

        self.f_beta = nn.Linear(DECODER_DIM, 1)

    def forward(self, encoder_out, h):
        # encoder_out: (B, L, encoder_dim)
        # h: (B, decoder_dim)
        att = self.W_a(encoder_out) + self.U_a(h).unsqueeze(1)  
        energy = self.w(torch.tanh(att)).squeeze(2)              
        alpha = torch.softmax(energy, dim=1)                   
        context = (encoder_out * alpha.unsqueeze(2)).sum(1)   
        
        beta = torch.sigmoid(self.f_beta(h))   
        context = beta * context  
        return context, alpha

# generates caption word by word using info from attention and previous words
class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD_IDX)
        self.attention = Attention()
        self.lstm = nn.LSTMCell(EMBED_DIM + ENCODER_DIM, DECODER_DIM)
        self.fc = nn.Linear(DECODER_DIM, vocab_size)
        self.dropout = nn.Dropout(DROPOUT)

        self.init_h = nn.Linear(ENCODER_DIM, DECODER_DIM)
        self.init_c = nn.Linear(ENCODER_DIM, DECODER_DIM)

    # initial hidden and mem states are predicted from the mean of the learned annotation vectors
    def init_hidden(self, encoder_out):
        mean = encoder_out.mean(dim=1)
        h = torch.tanh(self.init_h(mean))
        c = torch.tanh(self.init_c(mean))
        return h, c

    def forward(self, encoder_out, captions, lengths):
        lengths, sort_idx = lengths.sort(descending=True)
        encoder_out = encoder_out[sort_idx]
        captions = captions[sort_idx]

        embeddings = self.embedding(captions)  # embed captions
        h, c = self.init_hidden(encoder_out)

        # how many steps each sample needs,(lengths - 1) bc the last input is the word before <end>
        decode_lengths = (lengths - 1).tolist()
        max_t = int(max(decode_lengths))

        B = encoder_out.size(0)
        L = encoder_out.size(1)
        predictions = torch.zeros(B, max_t, self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(B, max_t, L).to(encoder_out.device)

        for t in range(max_t):
            # only process samples that still have tokens at step t
            Bt = sum(dl > t for dl in decode_lengths)

            context, alpha = self.attention(encoder_out[:Bt], h[:Bt]) # calc attention given curr hidden state
            h_new, c_new = self.lstm(
                torch.cat([embeddings[:Bt, t], context], dim=1), 
                (h[:Bt], c[:Bt])
            )

            predictions[:Bt, t] = self.fc(self.dropout(h_new))
            alphas[:Bt, t] = alpha

            h = h.clone()
            c = c.clone() 
            h[:Bt] = h_new
            c[:Bt] = c_new

        return predictions, captions, decode_lengths, alphas, sort_idx

    @torch.no_grad()
    def generate(self, encoder_out, max_len=MAX_CAPTION_LENGTH):
        # decoding for a single image
        device = encoder_out.device
        h, c = self.init_hidden(encoder_out)

        word = torch.tensor([START_IDX], device=device)
        caption = [START_IDX]
        alphas = []

        for _ in range(max_len):
            emb = self.embedding(word)           
            context, alpha = self.attention(encoder_out, h)
            h, c = self.lstm(torch.cat([emb, context], dim=1), (h, c))
            word = self.fc(h).argmax(1)          

            caption.append(word.item())
            alphas.append(alpha.squeeze(0).cpu())

            if word.item() == END_IDX:
                break

        return caption, alphas

def test():
    train_loader, _, _, vocab = get_loaders()
    images, captions, lengths, _ = next(iter(train_loader))

    encoder = EncoderCNN()
    decoder = DecoderLSTM(len(vocab))

    encoder_out = encoder(images)
    print("encoder_out:", encoder_out.shape)   # expect (B, 196, 512)

    predictions, caps_sorted, decode_lengths, alphas, sort_idx = decoder(encoder_out, captions, lengths)
    print("predictions:", predictions.shape)   # (B, max_t, vocab_size)
    print("alphas:", alphas.shape)             # (B, max_t, 196)
    print("decode_lengths[:5]:", decode_lengths[:5])

if __name__ == "__main__":
    test()