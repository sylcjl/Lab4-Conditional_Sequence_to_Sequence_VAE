#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import os
import unicodedata
import string
import re
from datetime import datetime
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from datasets import VocabTenseDataset

"""========================================================================================
The sample.py includes the following template functions:

1. Encoder, decoder
2. Training function
3. BLEU-4 score function
4. Gaussian score function

You have to modify them to complete the lab.
In addition, there are still other functions that you have to 
implement by yourself.

1. The reparameterization trick
2. Your own dataloader (design in your own way, not necessary Pytorch Dataloader)
3. Output your results (BLEU-4 score, conversion words, Gaussian score, generation words)
4. Plot loss/score
5. Load/save weights

There are some useful tips listed in the lab assignment.
You should check them before starting your lab.
========================================================================================"""
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
# ----------Hyper Parameters----------#
hidden_size = 256
latent_size = 32
# The number of vocabulary
vocab_size = 28
# The number of condition
condition_size = 4
condition_embedding_size = 8

n_iters = 300000
print_every = 30
plot_every = 100
save_every = 10000

adjust_iter_teacher_forcing = 75000
teacher_forcing_ratio = 1.0
lowest_teacher_forcing_ratio = 0.7
diff_teacher_forcing = (teacher_forcing_ratio - lowest_teacher_forcing_ratio) / (n_iters - adjust_iter_teacher_forcing)

empty_input_ratio = 0.1


KLD_weight = 0.0
highest_KLD_weight = 0.5
adjust_iter_KLD_weight = 10000
diff_KLD_weight = (highest_KLD_weight - KLD_weight) / (n_iters - adjust_iter_KLD_weight)

LR = 0.05
MAX_LENGTH = 30

time_now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
words_list = []

################################
# Example inputs of compute_bleu
################################
# The target word
reference = 'accessed'
# The word generated by your model
output = 'access'


# compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33, 0.33, 0.33)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu([reference], output, weights=weights, smoothing_function=cc.method1)


"""============================================================================
example input of Gaussian_score

words = [['consult', 'consults', 'consulting', 'consulted'],
['plead', 'pleads', 'pleading', 'pleaded'],
['explain', 'explains', 'explaining', 'explained'],
['amuse', 'amuses', 'amusing', 'amused'], ....]

the order should be : simple present, third person, present progressive, past
============================================================================"""
def load_words_list(path=os.path.join("lab4_dataset", "train.txt")):
    with open(path, 'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
    return words_list


def Gaussian_score(words):
    global words_list
    if not words_list:
        words_list = load_words_list(os.path.join("lab4_dataset", "train.txt"))

    score = 0
    for t in words:
        for i in words_list:
            if t == i:
                score += 1
    return score / len(words)


def adjust_teacher_forcing(iter):
    global teacher_forcing_ratio

    if iter > adjust_iter_teacher_forcing:
        teacher_forcing_ratio -= diff_teacher_forcing


def adjust_kld_weight(iter):
    global KLD_weight

    if iter > adjust_iter_KLD_weight:
        KLD_weight += diff_KLD_weight


#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, condition_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        hidden_condition_size = hidden_size + condition_size

        self.embedding = nn.Embedding(vocab_size, hidden_condition_size)
        self.rnn = nn.LSTM(hidden_condition_size, hidden_condition_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        hidden = torch.zeros(1, 1, self.hidden_size, device=device)
        cell = torch.zeros(1, 1, self.hidden_size, device=device)
        return hidden, hidden


#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        hidden = torch.zeros(1, 1, self.hidden_size, device=device)
        cell = torch.zeros(1, 1, self.hidden_size, device=device)
        return hidden, cell


class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def show_plot(results_dct, title):
    fig, ax = plt.subplots()
    ax.set_position([0, 0, 0.8, 1])
    ax2 = ax.twinx()
    plt.title(title)

    for kpi, values in results_dct.items():
        if not values:
            continue

        if kpi.startswith("loss"):
            ax.plot(values, "-", label=kpi)
        elif kpi.startswith("score_"):
            ax2.plot(values, "o", label=kpi)
        elif kpi.startswith("weight_"):
            ax2.plot(values, "--", label=kpi)
        else:
            pass

    figs = [i.get_legend_handles_labels() for i in (ax, ax2)]
    ax.legend(figs[0][0] + figs[1][0], figs[0][1] + figs[1][1], bbox_to_anchor=(1.02, 0.8), loc='center left')
    ax.set_xlabel('{} iterations'.format(plot_every))
    plt.savefig(os.path.join("results", time_now+"_"+title) + '.png')
    plt.clf()


def show_loss_plot(cross_entropy, kld, total, title):
    plt.title(title)
    plt.xlabel('Epoch')

    plt.plot(cross_entropy, label="Cross-Entropy")
    plt.plot(kld, label="KL Divergence")
    plt.plot(total, label="Total")

    plt.legend()
    plt.savefig(os.path.join("results", time_now+"_"+title) + '.png')
    plt.clf()


def show_bleu_plot(points, title):
    plt.title(title)
    plt.xlabel('Epoch')
    plt.plot(points, label="BLEU-4")
    plt.legend()
    plt.savefig(os.path.join("results", time_now+"_"+title) + '.png')
    plt.clf()


def train(input_tensor, target_tensor, embedding_condition,
          encoder, decoder, encoder_optimizer, decoder_optimizer,
          mean_layer_h, log_var_layer_h, latent_2_hidden_h,
          mean_optimizer_h, log_var_optimizer_h, lat2hidden_optimizer_h,
          mean_layer_c, log_var_layer_c, latent_2_hidden_c,
          mean_optimizer_c, log_var_optimizer_c, lat2hidden_optimizer_c,
          criterion, criterion_kld, max_length=MAX_LENGTH):

    input_vocab = input_tensor.vocab
    input_tense = input_tensor.tense
    target_vocab = target_tensor.vocab[:-1]
    target_tense = target_tensor.tense

    input_length = input_vocab.size(0)
    target_length = target_vocab.size(0)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    mean_optimizer_h.zero_grad()
    log_var_optimizer_h.zero_grad()
    lat2hidden_optimizer_h.zero_grad()
    mean_optimizer_c.zero_grad()
    log_var_optimizer_c.zero_grad()
    lat2hidden_optimizer_c.zero_grad()

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # ----------sequence to sequence part for encoder----------#
    condition_embedding = embedding_condition(input_tense).view(1, 1, -1)
    encoder_init_hidden, encoder_init_cell = encoder.initHidden()
    encoder_init_hidden = torch.cat([encoder_init_hidden, condition_embedding], dim=2)
    encoder_init_cell = torch.cat([encoder_init_cell, condition_embedding], dim=2)
    encoder_hidden = (encoder_init_hidden, encoder_init_cell)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_vocab[ei], encoder_hidden)

    # ---------- to latent space ----------#
    mean_h = mean_layer_h(encoder_hidden[0])
    logvar_h = log_var_layer_h(encoder_hidden[0])
    latent_tensor_h = reparameterize(mean_h, logvar_h)
    mean_c = mean_layer_h(encoder_hidden[1])
    logvar_c = log_var_layer_h(encoder_hidden[1])
    latent_tensor_c = reparameterize(mean_c, logvar_c)

    # ----------sequence to sequence part for decoder----------#
    decoder_outputs = []
    decoder_input = torch.tensor([SOS_token], dtype=torch.long, device=device)
    condition_embedding = embedding_condition(target_tense).view(1, 1, -1)
    encoder_init_hidden = torch.cat([latent_tensor_h, condition_embedding], dim=2)
    encoder_init_hidden = latent_2_hidden_h(encoder_init_hidden)
    encoder_init_cell = torch.cat([latent_tensor_c, condition_embedding], dim=2)
    encoder_init_cell = latent_2_hidden_c(encoder_init_cell)
    decoder_hidden = (encoder_init_hidden, encoder_init_cell)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # ----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)

            decoder_outputs.append(topi.squeeze().detach())

            loss += criterion(decoder_output, target_vocab[di].unsqueeze(0))
            decoder_input = target_vocab[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            decoder_outputs.append(decoder_input)

            loss += criterion(decoder_output, target_vocab[di].unsqueeze(0))
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    loss = loss.item() / target_length
    kld_loss_h = criterion_kld(mean_h, logvar_h).item() * KLD_weight
    total_loss = loss + kld_loss_h

    encoder_optimizer.step()
    decoder_optimizer.step()
    mean_optimizer_h.step()
    log_var_optimizer_h.step()
    lat2hidden_optimizer_h.step()
    mean_optimizer_c.step()
    log_var_optimizer_c.step()
    lat2hidden_optimizer_c.step()

    return decoder_outputs, total_loss, loss, kld_loss_h


def trainIters(encoder, decoder,
               mean_layer_h, log_var_layer_h, latent_2_hidden_h,
               mean_layer_c, log_var_layer_c, latent_2_hidden_c,
               embedding_condition,
               n_iters, print_every=1000, plot_every=100, save_every=2000, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    plot_cross_entropy_losses = []
    plot_kl_loss_losses = []

    results_dct = {
        "loss": [],
        "loss_ce": [],
        "loss_kld": [],
        "score_bleu4": [],
        "score_gaussian": [],
        "weight_teacher_forcing": [],
        "weight_KLD": [],
    }
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_cross_entropy_loss_total = 0
    plot_cross_entropy_loss_total = 0
    print_kl_loss_total = 0
    plot_kl_loss_total = 0
    print_bleu_total = 0
    plot_bleu_total = 0
    print_gaussian_total = 0
    plot_gaussian_total = 0
    plot_test_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    mean_optimizer_h = optim.SGD(mean_layer_h.parameters(), lr=learning_rate)
    log_var_optimizer_h = optim.SGD(log_var_layer_h.parameters(), lr=learning_rate)
    lat2hidden_optimizer_h = optim.SGD(latent_2_hidden_h.parameters(), lr=learning_rate)
    mean_optimizer_c = optim.SGD(mean_layer_c.parameters(), lr=learning_rate)
    log_var_optimizer_c = optim.SGD(log_var_layer_c.parameters(), lr=learning_rate)
    lat2hidden_optimizer_c = optim.SGD(latent_2_hidden_c.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)


    dataset = VocabTenseDataset(os.path.join("lab4_dataset", "train.txt"), device=device, SOS=SOS_token, EOS=EOS_token)
    pairs = dataset.embedded_pairs

    training_pairs = [random.choice(pairs) for i in range(n_iters)]
    criterion = nn.CrossEntropyLoss()
    criterion_kld = KLDivergenceLoss()

    for iter in range(1, n_iters + 1):

        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        output, loss, cross_entropy_loss, kl_loss = train(input_tensor, target_tensor, embedding_condition,
                                                          encoder, decoder, encoder_optimizer, decoder_optimizer,
                                                          mean_layer_h, log_var_layer_h, latent_2_hidden_h,
                                                          mean_optimizer_h, log_var_optimizer_h, lat2hidden_optimizer_h,
                                                          mean_layer_c, log_var_layer_c, latent_2_hidden_c,
                                                          mean_optimizer_c, log_var_optimizer_c, lat2hidden_optimizer_c,
                                                          criterion, criterion_kld)

        print_loss_total += loss
        plot_loss_total += loss
        print_cross_entropy_loss_total += cross_entropy_loss
        plot_cross_entropy_loss_total += cross_entropy_loss
        print_kl_loss_total += kl_loss
        plot_kl_loss_total += kl_loss

        input_vocab = dataset.to_vocab(input_tensor.vocab)
        target_vocab = dataset.to_vocab(target_tensor.vocab)
        output_vocab = dataset.to_vocab(output)
        bleu_score = compute_bleu(output_vocab, target_vocab)
        print_bleu_total += bleu_score
        plot_bleu_total += bleu_score
        gaussian_score = Gaussian_score(output_vocab)
        print_gaussian_total += gaussian_score


        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_cross_entropy_loss_avg = print_cross_entropy_loss_total / print_every
            print_cross_entropy_loss_total = 0
            print_kl_loss_avg = print_kl_loss_total / print_every
            print_kl_loss_total = 0
            print_bleu_avg = print_bleu_total / print_every
            print_bleu_total = 0

            print("Input: {}\toutput:{}\ttarget:{}".format(input_vocab, output_vocab, target_vocab))
            print('%s (%d %d%%) Loss=%.4f, BLEU-4=%.4f' % (timeSince(start, iter / n_iters),
                                                           iter, iter / n_iters * 100, print_loss_avg, print_bleu_avg))


        if iter % plot_every == 0:

            plot_loss_avg = plot_loss_total / plot_every
            results_dct["loss"].append(plot_loss_avg)
            plot_loss_total = 0

            plot_cross_entropy_loss_avg = plot_cross_entropy_loss_total / plot_every
            results_dct["loss_ce"].append(plot_cross_entropy_loss_avg)
            plot_cross_entropy_loss_total = 0

            plot_kl_loss_avg = plot_kl_loss_total / plot_every
            results_dct["loss_kld"].append(plot_kl_loss_avg)
            plot_kl_loss_total = 0

            plot_bleu_avg = plot_bleu_total / plot_every
            results_dct["score_bleu4"].append(plot_bleu_avg)
            plot_bleu_total = 0

            results_dct["weight_teacher_forcing"].append(teacher_forcing_ratio)
            results_dct["weight_KLD"].append(KLD_weight)

            show_plot(results_dct, title="Training KPI trends")

        if iter % save_every == 0:
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "embedding_condition": embedding_condition.state_dict(),
                "mean_layer_h": mean_layer_h.state_dict(),
                "log_var_layer_h": log_var_layer_h.state_dict(),
                "latent_2_hidden_h": latent_2_hidden_h.state_dict(),
                "mean_layer_c": mean_layer_c.state_dict(),
                "log_var_layer_c": log_var_layer_c.state_dict(),
                "latent_2_hidden_c": latent_2_hidden_c.state_dict(),
                "teacher_forcing_ratio": teacher_forcing_ratio,
                "KLD_weight": KLD_weight,
            }, 'results/{}-Model-{}.ckpt'.format(time_now, iter))

        adjust_teacher_forcing(iter)
        adjust_kld_weight(iter)


embedding_condition = nn.Embedding(condition_size, condition_embedding_size).to(device)
encoder1 = EncoderRNN(vocab_size, hidden_size, condition_embedding_size).to(device)
mean_layer_h = nn.Linear(hidden_size + condition_embedding_size, latent_size).to(device)
log_var_layer_h = nn.Linear(hidden_size + condition_embedding_size, latent_size).to(device)
latent_2_hidden_h = nn.Linear(latent_size + condition_embedding_size, hidden_size).to(device)
mean_layer_c = nn.Linear(hidden_size + condition_embedding_size, latent_size).to(device)
log_var_layer_c = nn.Linear(hidden_size + condition_embedding_size, latent_size).to(device)
latent_2_hidden_c = nn.Linear(latent_size + condition_embedding_size, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, vocab_size).to(device)

trainIters(encoder1, decoder1,
           mean_layer_h, log_var_layer_h, latent_2_hidden_h,
           mean_layer_c, log_var_layer_c, latent_2_hidden_c,
           embedding_condition,
           n_iters, print_every=print_every, plot_every=plot_every, save_every=save_every)
