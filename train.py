import torch
import torch.nn as nn
import time
from torch.autograd import Variable
from torch import optim
from hp import PAD_token, SOS_token, EOS_token, MIN_LENGTH, MAX_LENGTH
from masked_cross_entropy import *
import random
import math
from hp import teacher_forcing_ratio, clip, batch_size, learning_rate

use_cuda = torch.cuda.is_available()


# Pad a with the PAD symbol# Pad a
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 5000))
    print("Adjusted learning rate to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length=MAX_LENGTH, batch_size=batch_size):
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if use_cuda:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t]  # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths
    )
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0], ec, dc


def train_iters(encoder, decoder, input_lang, output_lang, pairs, n_epochs=50000, print_every=100, evaluate_every=100,
                learning_rate=learning_rate,
                decoder_learning_ratio=5.0, batch_size=50):
    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    print("Starting training for n_epochs={} lr={}, batch_size={}".format(n_epochs, learning_rate, batch_size))

    for epoch in range(1, n_epochs + 1):

        if epoch % 1000 == 0:
            adjust_learning_rate(decoder_optimizer, epoch)

        batch_it = batch(pairs, n=batch_size)
        num_iters = math.ceil(len(pairs) / batch_size)

        for i in range(num_iters):

            # Get training data for this cycle
            input_batches, input_lengths, target_batches, target_lengths = next_batch(input_lang, output_lang, batch_it,
                                                                                      batch_size)

            # Run the train function
            loss, ec, dc = train(
                input_batches, input_lengths, target_batches, target_lengths,
                encoder, decoder,
                encoder_optimizer, decoder_optimizer, criterion, batch_size=batch_size
            )

            # Keep track of loss
            print_loss_total += loss
            plot_loss_total += loss

            if i % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print_summary = '%s (%d %d%% %d%%) %.4f' % (
                    time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, i / num_iters * 100,
                    print_loss_avg)
                print(print_summary)

            if epoch % evaluate_every == 0:
                pass
                # evaluate_randomly()


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def next_batch(input_lang, output_lang, batch_it, batch_size):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for pair in next(batch_it):
        input_seqs.append(indexes_from_sentence(input_lang, pair[0]))
        target_seqs.append(indexes_from_sentence(output_lang, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if use_cuda:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, input_lengths, target_var, target_lengths


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))
