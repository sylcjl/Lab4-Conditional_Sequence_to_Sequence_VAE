#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import random
import torch.nn as nn
import torch.nn.functional as F


"""
https://github.com/pytorch/examples/blob/master/vae/main.py
https://github.com/wiseodd/generative-models/blob/master/VAE/conditional_vae/cvae_pytorch.py
"""

class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())


#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, condition_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        hidden_condition_size = hidden_size + condition_size

        self.embedding = nn.Embedding(vocab_size, hidden_condition_size)
        self.rnn = nn.LSTM(hidden_condition_size, hidden_condition_size)

        self.init_hidden = torch.zeros(1, 1, self.hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        hidden = self.init_hidden.clone()
        cell = self.init_hidden.clone()
        return hidden, cell


#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.init_hidden = torch.zeros(1, 1, self.hidden_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        hidden = self.init_hidden.clone()
        cell = self.init_hidden.clone()
        return hidden, cell


class CVAE(nn.Module):
    def __init__(self, vocab_size, latent_size, hidden_size, condition_size, condition_embedding_size,
                 teacher_forcing_ratio=1.0,
                 SOS=0, EOS=1, device=None):
        super(CVAE, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = EncoderRNN(vocab_size, hidden_size, condition_embedding_size).to(self.device)
        self.decoder = DecoderRNN(hidden_size, vocab_size).to(self.device)
        self.embedding_condition = nn.Embedding(condition_size, condition_embedding_size)

        self.h_mean_layer = nn.Linear(hidden_size + condition_embedding_size, latent_size)
        self.h_logvar_layer = nn.Linear(hidden_size + condition_embedding_size, latent_size)
        self.h_latent_2_hidden = nn.Linear(latent_size + condition_embedding_size, hidden_size)

        self.c_mean_layer = nn.Linear(hidden_size + condition_embedding_size, latent_size)
        self.c_logvar_layer = nn.Linear(hidden_size + condition_embedding_size, latent_size)
        self.c_latent_2_hidden = nn.Linear(latent_size + condition_embedding_size, hidden_size)

        self.SOS = SOS
        self.EOS = EOS
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def encode(self, input_tensor):
        input_vocab = input_tensor.vocab
        input_tense = input_tensor.tense
        input_length = input_vocab.size(0)

        # ----------sequence to sequence part for encoder----------#
        condition_embedding = self.embedding_condition(input_tense).view(1, 1, -1)
        encoder_init_hidden, encoder_init_cell = self.encoder.initHidden()
        encoder_init_hidden = torch.cat([encoder_init_hidden.to(self.device), condition_embedding], dim=2)
        encoder_init_cell = torch.cat([encoder_init_cell.to(self.device), condition_embedding], dim=2)
        encoder_hidden = (encoder_init_hidden, encoder_init_cell)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_vocab[ei], encoder_hidden)

        # ---------- to latent space ----------#
        mean_h = self.h_mean_layer(encoder_hidden[0])
        logvar_h = self.h_logvar_layer(encoder_hidden[0])

        mean_c = self.c_mean_layer(encoder_hidden[1])
        logvar_c = self.c_logvar_layer(encoder_hidden[1])

        return (mean_h, logvar_h), (mean_c, logvar_c)

    def decode(self, target_tensor, latent_tensor):
        loss_count = 0
        cross_entropy_loss = 0
        decoder_outputs = []

        target_vocab = target_tensor.vocab[:-1]
        target_tense = target_tensor.tense
        target_length = target_vocab.size(0)

        decoder_input = torch.tensor([self.SOS], dtype=torch.long).to(self.device)
        condition_embedding = self.embedding_condition(target_tense).view(1, 1, -1)

        # Handle initial hidden layer
        h_decoder_init = torch.cat([latent_tensor[0].to(self.device), condition_embedding], dim=2)
        h_decoder_init = self.h_latent_2_hidden(h_decoder_init)
        c_decoder_init = torch.cat([latent_tensor[1].to(self.device), condition_embedding], dim=2)
        c_decoder_init = self.c_latent_2_hidden(c_decoder_init)
        decoder_hidden = (h_decoder_init, c_decoder_init)

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        # ----------sequence to sequence part for decoder----------#
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_outputs.append(topi.squeeze().detach())

                cross_entropy_loss += F.cross_entropy(decoder_output, target_vocab[di].unsqueeze(0))
                decoder_input = target_vocab[di]  # Teacher forcing
                loss_count += 1
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                decoder_outputs.append(decoder_input)

                cross_entropy_loss += F.cross_entropy(decoder_output, target_vocab[di].unsqueeze(0))
                loss_count += 1
                if decoder_input.item() == self.EOS_token:
                    break

        cross_entropy_loss /= loss_count

        return decoder_outputs, cross_entropy_loss

    def forward(self, input_tensor, target_tensor):
        h_mean_logvar, c_mean_logvar = self.encode(input_tensor=input_tensor)

        h_latent_tensor = self.reparameterize(*h_mean_logvar)
        c_latent_tensor = self.reparameterize(*c_mean_logvar)

        output, cross_entropy_loss = self.decode(target_tensor=target_tensor,
                                                 latent_tensor=(h_latent_tensor, c_latent_tensor))
        h_kl_divergence_loss = self.kl_divergence(*h_mean_logvar)
        c_kl_divergence_loss = self.kl_divergence(*c_mean_logvar)
        return output, cross_entropy_loss, (h_kl_divergence_loss, c_kl_divergence_loss)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
