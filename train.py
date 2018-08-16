import torch
import torch.nn as nn
import time
from torch.autograd import Variable
from torch import optim
from hp import PAD_token, SOS_token, EOS_token, MIN_LENGTH, MAX_LENGTH
from models import Seq2SeqModel
import random
import math
import pickle
import hp
from hp import teacher_forcing_ratio, clip, batch_size, learning_rate

use_cuda = torch.cuda.is_available()


# Pad a with the PAD symbol# Pad a
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 10))
    print("Adjusted learning rate to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_teacher_forcing(iter):
    return teacher_forcing_ratio * (0.9 ** ((iter - 1) // 20000))


def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length=MAX_LENGTH, batch_size=batch_size,
          teacher_forcing_ratio=teacher_forcing_ratio):
    # Zero gradients of both optimizers
    if encoder_optimizer:
        encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    decoder_hidden = encoder_hidden
    last_attn_vector = torch.zeros((batch_size, decoder.hidden_size))

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if use_cuda:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()
        last_attn_vector = last_attn_vector.cuda()

    teacher_force = random.random() < teacher_forcing_ratio

    if teacher_force:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn, last_attn_vector = decoder(
                decoder_input, decoder_hidden, encoder_outputs, last_attn_vector
            )
            loss += criterion(decoder_output, target_batches[t])
            all_decoder_outputs[t] = decoder_output
            decoder_input = target_batches[t]  # Next input is current target
    else:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn, last_attn_vector = decoder(
                decoder_input, decoder_hidden, encoder_outputs, last_attn_vector
            )

            all_decoder_outputs[t] = decoder_output
            v, i = decoder_output.topk(1)

            loss += criterion(decoder_output, target_batches[t])
            decoder_input = i.view(-1).detach()
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    if encoder_optimizer:
        encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / max_target_length, ec, dc


def train_iters(seq2seq_model, pairs,
                eval_pairs,
                n_epochs=hp.n_epochs,
                print_every=hp.print_loss_every_iters,
                evaluate_every=hp.eval_bleu_every_epochs,
                save_every=hp.save_every_epochs,
                learning_rate=hp.learning_rate,
                decoder_learning_ratio=hp.decoder_learning_ratio,
                batch_size=hp.batch_size,
                encoder_optimizer_state=None,
                decoder_optimizer_state=None, train_loss=[], eval_loss=[],
                bleu_scores=[],
                start_epoch=1,
                retrain=False,
                weight_decay=1e-5):
    encoder = seq2seq_model.encoder
    decoder = seq2seq_model.decoder
    input_lang = seq2seq_model.input_lang
    output_lang = seq2seq_model.output_lang

    # Initialize optimizers and criterion
    if not retrain:
        encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate,
                                       weight_decay=weight_decay)
    else:
        encoder_optimizer = None

    decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()),
                                   lr=learning_rate * decoder_learning_ratio, weight_decay=weight_decay)

    if encoder_optimizer_state:
        encoder_optimizer.load_state_dict(encoder_optimizer_state)
    if decoder_optimizer_state:
        decoder_optimizer.load_state_dict(decoder_optimizer_state)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    print("Starting training for n_epochs={} lr={}, batch_size={}".format(n_epochs, learning_rate, batch_size))
    print("Source File: {} Target File {}: Reverse: {}".format(hp.source_file, hp.target_file, hp.reverse_languages))

    for epoch in range(start_epoch, n_epochs + 1):

        # lr = adjust_learning_rate(decoder_optimizer, epoch)
        lr = learning_rate

        print_loss_total = 0
        epoch_training_loss = 0

        batch_it = batch(pairs, n=batch_size)
        num_iters = math.floor(len(pairs) / batch_size)

        for i in range(1, num_iters + 1):

            teacher_forcing_ratio = adjust_teacher_forcing(i)

            # Get training data for this cycle
            input_batches, input_lengths, target_batches, target_lengths = next_batch(input_lang, output_lang, batch_it,
                                                                                      batch_size)

            # Run the train function
            loss, ec, dc = train(
                input_batches, input_lengths, target_batches, target_lengths,
                encoder, decoder,
                encoder_optimizer, decoder_optimizer, criterion, batch_size=batch_size,
                teacher_forcing_ratio=teacher_forcing_ratio
            )

            # Keep track of loss
            print_loss_total += loss
            plot_loss_total += loss
            epoch_training_loss += loss

            if i % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print_summary = '%s (Epoch %d %d%% Iter %d %d%%) %.4f lr: %f tf: %f' % (
                    time_since(start, (epoch * i) / (n_epochs * num_iters)), epoch, epoch / n_epochs * 100, i,
                    i / num_iters * 100,
                    print_loss_avg, lr, teacher_forcing_ratio)
                print(print_summary)

        # Finished iterations
        avg_training_loss = epoch_training_loss / num_iters

        eval_batch_it = batch(eval_pairs, n=batch_size)
        eval_num_iters = math.floor(len(eval_pairs) / batch_size)

        epoch_evaluation_loss = 0
        for i in range(1, eval_num_iters + 1):
            input_batches, input_lengths, target_batches, target_lengths = next_batch(input_lang, output_lang,
                                                                                      eval_batch_it,
                                                                                      batch_size)
            curr_eval_loss = eval(input_batches, input_lengths, target_batches, target_lengths,
                                  encoder, decoder, criterion)
            epoch_evaluation_loss += curr_eval_loss

        if retrain:
            continue

        avg_evaluation_loss = epoch_evaluation_loss / eval_num_iters

        if epoch % evaluate_every == 0:
            bleu = seq2seq_model.eval_bleu()
            print("BLEU = {}".format(bleu))
            bleu_scores.append(bleu)

        train_loss.append(avg_training_loss)
        eval_loss.append(avg_evaluation_loss)

        if epoch % save_every == 0:
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "encoder_optimizer": encoder_optimizer.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "bleu_scores": bleu_scores
            }, hp.checkpoint_name)

        print("Avg. Training Loss: %.2f Avg. Evaluation Loss: %.2f" % (avg_training_loss, avg_evaluation_loss))


def eval(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, criterion):
    # Zero gradients of both optimizers
    encoder.train(False)
    decoder.train(False)
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    decoder_hidden = encoder_hidden
    last_attn_vector = torch.zeros((batch_size, decoder.hidden_size))

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if use_cuda:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()
        last_attn_vector = last_attn_vector.cuda()

    teacher_force = True

    if teacher_force:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn, last_attn_vector = decoder(
                decoder_input, decoder_hidden, encoder_outputs, last_attn_vector
            )
            loss += criterion(decoder_output, target_batches[t])
            all_decoder_outputs[t] = decoder_output
            decoder_input = target_batches[t]  # Next input is current target
    else:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn, last_attn_vector = decoder(
                decoder_input, decoder_hidden, encoder_outputs, last_attn_vector
            )

            all_decoder_outputs[t] = decoder_output
            v, i = decoder_output.topk(1)

            loss += criterion(decoder_output, target_batches[t])
            decoder_input = i.view(-1).detach()
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    encoder.train(True)
    decoder.train(True)
    return loss.item() / max_target_length


def retrain_iters(seq2seq_model, pairs,
                  eval_pairs,
                  n_epochs=hp.n_epochs,
                  print_every=hp.print_loss_every_iters,
                  evaluate_every=hp.eval_bleu_every_epochs,
                  save_every=hp.save_every_epochs,
                  learning_rate=hp.learning_rate,
                  decoder_learning_ratio=hp.decoder_learning_ratio,
                  batch_size=hp.batch_size,
                  weight_decay=1e-5):
    encoder, decoder = seq2seq_model.encoder, seq2seq_model.decoder

    encoder.train(True)
    decoder.train(True)

    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    for param in decoder.out.parameters():
        param.requires_grad = True

    train_iters(seq2seq_model, pairs, eval_pairs, n_epochs, print_every, evaluate_every,
                save_every, learning_rate, decoder_learning_ratio, batch_size, start_epoch=1, retrain=True)


def batch(iterable, n=1):
    l = len(iterable)
    random.shuffle(iterable)
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
