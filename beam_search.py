import torch
import torch.nn as nn
from torch.autograd import Variable

MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1


class Hypothesis:
    def __init__(self, tokens, log_probs, state, context, attns=None):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.attns = [[]] if attns is None else attns

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        return sum(self.log_probs)

    def extend(self, token, new_log_prob, new_state, context, attn):
        return Hypothesis(self.tokens + [token], self.log_probs + [new_log_prob], new_state, context,
                          self.attns + [attn])

    def __str__(self):
        return ('Hypothesis(log prob = %.4f, tokens = %s)' % (self.log_prob, self.tokens))

    def __repr__(self):
        return ('Hypothesis(log prob = %.4f, tokens = %s)' % (self.log_prob, self.tokens))


class BeamSearch:
    def __init__(self, decoder, encoder_outputs, decoder_hidden, output_lang,
                 beam_size=3, attention_override=None, partial=None, max_length=MAX_LENGTH):
        self.decoder = decoder
        self.encoder_outputs = encoder_outputs
        self.decoder_hidden = decoder_hidden
        self.beam_size = beam_size
        self.max_length = MAX_LENGTH
        self.output_lang = output_lang
        self.attention_override = attention_override
        self.partial = partial

    def decode_topk(self, latest_tokens, states, contexts, partials):

        # len(latest_tokens) x self.beam_size)
        topk_ids = [[0 for _ in range(self.beam_size)] for _ in range(len(latest_tokens))]
        topk_log_probs = [[0 for _ in range(self.beam_size)] for _ in range(len(latest_tokens))]
        new_states = [None] * len(states)
        new_contexts = [None] * len(states)
        attns = [None] * len(states)

        for token, state, context, i in zip(latest_tokens, states, contexts, range(len(latest_tokens))):
            decoder_input = Variable(torch.LongTensor([[token]]))

            attention_override = self.attention_override if self.partial == partials[i] else None

            print("Partial")
            print(self.partial)
            print(partials[i])
            decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, context,
                                                                                              state,
                                                                                              self.encoder_outputs,
                                                                                              attention_override)

            topk_v, topk_i = decoder_output.data.topk(self.beam_size)
            topk_v, topk_i = topk_v.numpy()[0], topk_i.numpy()[0]

            topk_ids[i] = topk_i.tolist()
            topk_log_probs[i] = topk_v.tolist()

            new_contexts[i] = decoder_context
            new_states[i] = decoder_hidden
            attns[i] = decoder_attention.data.numpy().tolist()[0]

        return topk_ids, topk_log_probs, new_states, new_contexts, attns

    def to_partial(self, tokens):
        return " ".join([self.output_lang.index2word[token] for token in tokens])

    def search(self):

        hyps = [Hypothesis([SOS_token], [0.0], self.decoder_hidden, Variable(torch.zeros(1, self.decoder.hidden_size)))
                for _ in range(self.beam_size)]
        result = []

        steps = 0

        while steps < self.max_length and len(result) < self.beam_size:
            latest_tokens = [hyp.latest_token for hyp in hyps]
            states = [hyp.state for hyp in hyps]
            contexts = [hyp.context for hyp in hyps]
            partials = [self.to_partial(hyp.tokens) for hyp in hyps]
            all_hyps = []

            num_beam_source = 1 if steps == 0 else len(hyps)
            topk_ids, topk_log_probs, new_states, contexts, attns = self.decode_topk(latest_tokens, states, contexts,
                                                                                     partials)

            for i in range(num_beam_source):
                h, ns, attn, context = hyps[i], new_states[i], attns[i], contexts[i]

                for j in range(self.beam_size):
                    all_hyps.append(h.extend(topk_ids[i][j], topk_log_probs[i][j], ns, context, attn))

            # Filter
            hyps = []

            for h in self._best_hyps(all_hyps):
                if h.latest_token == EOS_token:
                    result.append(h)
                else:
                    hyps.append(h)
                if len(hyps) == self.beam_size or len(result) == self.beam_size:
                    break
            print(hyps)
            steps += 1

        return self._best_hyps(result)

    def _best_hyps(self, hyps):
        """Sort the hyps based on log probs and length.
        Args:
          hyps: A list of hypothesis.
        Returns:
          hyps: A list of sorted hypothesis in reverse log_prob order.
        """
        return sorted(hyps, key=lambda h: h.log_prob, reverse=True)
