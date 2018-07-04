import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from hp import MAX_LENGTH, n_layers

use_cuda = torch.cuda.is_available()


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, n_layers=n_layers, dropout=0.1):
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
        return F.softmax(attn_energies).unsqueeze(1)

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
            v = self.v.repeat(B, 1).unsqueeze(1)
            energy = torch.bmm(v, energy)
            return energy.squeeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=n_layers, dropout=0.1):
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
