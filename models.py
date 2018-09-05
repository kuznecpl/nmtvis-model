import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from beam_search import BeamSearch
import numpy as np

from hp import MAX_LENGTH, n_layers, SOS_token, EOS_token, PAD_token, UNK_token
import hp

use_cuda = torch.cuda.is_available()


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, n_layers=n_layers, dropout=hp.dropout):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over the whole input sequence)
        if use_cuda and hidden: hidden = hidden.cuda()

        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs

        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size, max_length=MAX_LENGTH):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len))  # B x S

        if use_cuda:
            attn_energies = attn_energies.cuda()

        attn_energies = self.score(hidden, encoder_outputs)

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.permute([1, 0, 2]).bmm(encoder_output.permute([1, 2, 0]))
            return energy.squeeze(1)

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.permute([1, 0, 2]).bmm(energy.permute([1, 2, 0]))  # B x 1 x H bmm B x H x S => B x 1 x S
            return energy.squeeze(1)  # B X S

        elif self.method == 'concat':
            S = encoder_output.size(0)
            B = encoder_output.size(1)

            hidden = hidden.repeat(S, 1, 1).transpose(0, 1)  # 1 x B x H => B x S x H
            concat = torch.cat((hidden, encoder_output.transpose(0, 1)), 2)  # B x S x 2H

            energy = self.attn(concat).transpose(2, 1)  # B x H x S
            energy = F.tanh(energy)
            v = self.v.repeat(B, 1).unsqueeze(1)
            energy = torch.bmm(v, energy)
            return energy.squeeze(1)


class LSTMEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, n_layers=n_layers, dropout=hp.dropout, bidirectional=True):
        super(LSTMEncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = self.hidden_size // self.num_directions

        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, self.hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over the whole input sequence)
        if use_cuda and hidden: hidden = hidden.cuda()

        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)

        if self.bidirectional:
            # (num_layers * num_directions, batch_size, hidden_size)
            # => (num_layers, batch_size, hidden_size * num_directions)
            hidden = self._cat_directions(hidden)

        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs

        return outputs, hidden

    def _cat_directions(self, hidden):
        """ If the encoder is bidirectional, do the following transformation.
            Ref: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py#L176
            -----------------------------------------------------------
            In: (num_layers * num_directions, batch_size, hidden_size)
            (ex: num_layers=2, num_directions=2)

            layer 1: forward__hidden(1)
            layer 1: backward_hidden(1)
            layer 2: forward__hidden(2)
            layer 2: backward_hidden(2)

            -----------------------------------------------------------
            Out: (num_layers, batch_size, hidden_size * num_directions)

            layer 1: forward__hidden(1) backward_hidden(1)
            layer 2: forward__hidden(2) backward_hidden(2)
        """

        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        if isinstance(hidden, tuple):
            # LSTM hidden contains a tuple (hidden state, cell state)
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU hidden
            hidden = _cat(hidden)

        return hidden


class LSTMAttnDecoderRNN(nn.Module):
    def __init__(self, encoder, attn_model, hidden_size, output_size, n_layers=n_layers, dropout=hp.dropout):
        super(LSTMAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = encoder.hidden_size * encoder.num_directions
        self.output_size = output_size
        self.n_layers = encoder.n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, self.hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, self.hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, last_attn_vector, attention_override=None):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)

        # B x O => B x H
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)

        # Input feeding
        # B x H cat B x H => B x 2H
        embedded = torch.cat((embedded, last_attn_vector), dim=1)

        embedded = embedded.view(1, batch_size, 2 * self.hidden_size)  # S=1 x B x 2H

        # rnn_output: 1 x B x H, hidden: 1 x B x H
        rnn_output, hidden = self.lstm(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        if attention_override is not None:
            attn_weights = Variable(torch.FloatTensor(attention_override + [0]).view(1, 1, len(attention_override) + 1))

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x H -> B x H
        context = context.squeeze(1)  # B x S=1 x N -> B x H
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))  # B x H

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights, concat_output


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=n_layers, dropout=hp.dropout):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, attention_override=None):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size)  # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        if attention_override is not None:
            attn_weights = Variable(torch.FloatTensor(attention_override + [0]).view(1, 1, len(attention_override) + 1))

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights


class LSTMEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, n_layers=n_layers, dropout=hp.dropout, bidirectional=True):
        super(LSTMEncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = self.hidden_size // self.num_directions

        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, self.hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over the whole input sequence)
        if use_cuda and hidden: hidden = hidden.cuda()

        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)

        if self.bidirectional:
            # (num_layers * num_directions, batch_size, hidden_size)
            # => (num_layers, batch_size, hidden_size * num_directions)
            hidden = self._cat_directions(hidden)

        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs

        return outputs, hidden

    def _cat_directions(self, hidden):
        """ If the encoder is bidirectional, do the following transformation.
            Ref: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py#L176
            -----------------------------------------------------------
            In: (num_layers * num_directions, batch_size, hidden_size)
            (ex: num_layers=2, num_directions=2)

            layer 1: forward__hidden(1)
            layer 1: backward_hidden(1)
            layer 2: forward__hidden(2)
            layer 2: backward_hidden(2)

            -----------------------------------------------------------
            Out: (num_layers, batch_size, hidden_size * num_directions)

            layer 1: forward__hidden(1) backward_hidden(1)
            layer 2: forward__hidden(2) backward_hidden(2)
        """

        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        if isinstance(hidden, tuple):
            # LSTM hidden contains a tuple (hidden state, cell state)
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU hidden
            hidden = _cat(hidden)

        return hidden


class LSTMAttnDecoderRNN(nn.Module):
    def __init__(self, encoder, attn_model, hidden_size, output_size, n_layers=n_layers, dropout=hp.dropout):
        super(LSTMAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = encoder.hidden_size * encoder.num_directions
        self.output_size = output_size
        self.n_layers = encoder.n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, self.hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, self.hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, last_attn_vector, attention_override=None):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)

        # B x O => B x H
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)

        # Input feeding
        # B x H cat B x H => B x 2H
        embedded = torch.cat((embedded, last_attn_vector), dim=1)

        embedded = embedded.view(1, batch_size, 2 * self.hidden_size)  # S=1 x B x 2H

        # rnn_output: 1 x B x H, hidden: 1 x B x H
        rnn_output, hidden = self.lstm(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        if attention_override is not None:
            attn_weights = Variable(torch.FloatTensor(attention_override + [0]).view(1, 1, len(attention_override) + 1))

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x H -> B x H
        context = context.squeeze(1)  # B x S=1 x N -> B x H
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))  # B x H

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights, concat_output


class Seq2SeqModel:
    def __init__(self, encoder, decoder, input_lang, output_lang):
        self.encoder = encoder
        self.decoder = decoder
        self.input_lang = input_lang
        self.output_lang = output_lang

    def pad_seq(self, seq, max_length):
        seq += [PAD_token for i in range(max_length - len(seq))]
        return seq

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def next_batch(self, batch_it):
        input_seqs = []

        # Choose random pairs
        for pair in batch_it:
            input_seqs.append(indexes_from_sentence(self.input_lang, pair))

        indices = [i for i in range(len(batch_it))]
        # Zip into pairs, sort by length (descending), unzip
        input_seqs, indices = zip(*sorted(zip(input_seqs, indices), key=lambda p: len(p[0]), reverse=True))

        # For input and target sequences, get array of lengths and pad with 0s to max length
        input_lengths = [len(s) for s in input_seqs]
        input_padded = [self.pad_seq(s, max(input_lengths)) for s in input_seqs]

        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)

        if use_cuda:
            input_var = input_var.cuda()

        return input_var, input_lengths, indices

    def idx_to_sentence(self, idx):
        eos_id = self.output_lang.word2index[hp.EOS_text]
        eos = idx.index(eos_id) if eos_id in idx else len(idx) - 1
        return " ".join(self.output_lang.index2word[i] for i in idx[:eos + 1])

    def batch_translate(self, sentences):
        translations = []
        for sentence_batch in self.batch(sentences, min(hp.batch_size, len(sentences))):
            input_batch, input_lengths, sort_indices = self.next_batch(sentence_batch)
            outputs = self._batch_translate(input_batch, input_lengths, 50, len(sentence_batch))
            idx = outputs.topk(1, dim=2)[1].squeeze(2).transpose(1, 0).cpu().numpy().tolist()
            sorted_translations = [self.idx_to_sentence(i) for i in idx]
            batch_translations = [None] * len(sentence_batch)
            for i, t in zip(sort_indices, sorted_translations):
                batch_translations[i] = t
            translations += batch_translations
        return translations

    def _batch_translate(self, input_batches, input_lengths, max_length=MAX_LENGTH, batch_size=256):
        if use_cuda:
            input_batches = input_batches.cuda()

        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        
        # Run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
        decoder_hidden = encoder_hidden
        last_attn_vector = torch.zeros((batch_size, self.decoder.hidden_size))
        all_decoder_outputs = Variable(torch.zeros(max_length, batch_size, self.decoder.output_size))

        # Move new Variables to CUDA
        if use_cuda:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()
            last_attn_vector = last_attn_vector.cuda()

        # Run through decoder one time step at a time
        for t in range(max_length):
            decoder_output, decoder_hidden, decoder_attn, last_attn_vector = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, last_attn_vector
            )

            all_decoder_outputs[t] = decoder_output
            v, i = decoder_output.topk(1)
            decoder_input = i.view(-1)
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        self.encoder.train(True)
        self.decoder.train(True)
        return all_decoder_outputs

    def neat_translation(self, sentence, beam_size=1):
        hyps = self.beam_search(sentence, beam_size)
        words, attention = self.best_translation(hyps)

        return " ".join(words), np.matrix(attention)

    def best_translation(self, hyps):
        translation = [self.output_lang.index2word[token] for token in hyps[0].tokens] if hyps else ["SOS"]
        attn = [attn[0] for attn in hyps[0].attns] if hyps else [0]

        return translation[1:], attn[1:]

    def translate(self, sentence, beam_size=3, beam_length=0.5, beam_coverage=0.5, attention_override_map=None,
                  correction_map=None, unk_map=None, max_length=MAX_LENGTH):

        hyps = self.beam_search(sentence, beam_size, attention_override_map, correction_map, unk_map,
                                beam_length=beam_length, beam_coverage=beam_coverage, max_length=max_length)
        words, attention = self.best_translation(hyps)

        return words, attention, [Translation.from_hypothesis(h, self.output_lang) for h in hyps]

    def beam_search(self, input_seq, beam_size=3, attentionOverrideMap=None, correctionMap=None, unk_map=None,
                    beam_length=0.5, beam_coverage=0.5,
                    max_length=MAX_LENGTH):

        torch.set_grad_enabled(False)

        input_seqs = [indexes_from_sentence(self.input_lang, input_seq)]
        input_lengths = [len(seq) for seq in input_seqs]
        input_batches = Variable(torch.LongTensor(input_seqs)).transpose(0, 1)

        if use_cuda:
            input_batches = input_batches.cuda()

        self.encoder.train(False)
        self.decoder.train(False)

        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)

        decoder_hidden = encoder_hidden

        beam_search = BeamSearch(self.decoder, encoder_outputs, decoder_hidden, self.output_lang, beam_size,
                                 attentionOverrideMap,
                                 correctionMap, unk_map, beam_length=beam_length, beam_coverage=beam_coverage)
        result = beam_search.search()

        self.encoder.train(True)
        self.decoder.train(True)

        torch.set_grad_enabled(True)

        return result  # Return a list of indexes, one for each word in the sentence, plus EOS

    def eval_bleu(self, source_test_file=hp.source_test_file, target_test_file=hp.target_test_file):
        self.encoder.train(False)
        self.decoder.train(False)

        print("Evaluating BLEU")

        if hp.reverse_languages:
            source_test_file, target_test_file = target_test_file, source_test_file

        references = []

        with open(target_test_file, "r") as f:
            for line in f.readlines():
                references.append([line.strip().split(" ")])

        references = references

        source_sentences = []
        with open(source_test_file, "r") as f:
            for line in f.readlines():
                source_sentences.append(line.strip())

        source_sentences = source_sentences

        translations = []
        for sentence in source_sentences:
            translation, _, _ = self.translate(sentence, max_length=220)
            translations.append(translation[:-1])

        import nltk
        bleu = nltk.translate.bleu_score.corpus_bleu(references, translations)

        self.encoder.train(True)
        self.decoder.train(True)

        return bleu


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]


class Translation:
    def __init__(self, words=None, log_probs=None, attns=None, candidates=None, is_golden=False, is_unk=None):
        self.words = words
        self.log_probs = log_probs
        self.attns = attns
        self.candidates = candidates
        self.is_golden = is_golden
        self.is_unk = is_unk

    def slice(self):
        return Translation(self.words[1:], self.log_probs[1:], self.attns[1:], self.candidates[1:], self.is_golden,
                           self.is_unk[1:])

    @classmethod
    def from_hypothesis(cls, hypothesis, output_lang):
        translation = Translation()

        translation.words = hypothesis.words
        translation.log_probs = hypothesis.log_probs
        translation.attns = hypothesis.attns
        translation.candidates = hypothesis.candidates
        translation.is_golden = hypothesis.is_golden
        translation.is_unk = hypothesis.is_unk

        return translation
