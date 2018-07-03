import torch
import torch.nn as nn
from torch.autograd import Variable

MAX_LENGTH = 10
SOS_token = 1
EOS_token = 2


class Hypothesis:
    def __init__(self, tokens, log_probs, state, attns=None, candidates=None):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attns = [[]] if attns is None else attns
        # candidate tokens at each search step
        self.candidates = candidates

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        return sum(self.log_probs)

    def extend(self, token, new_log_prob, new_state, attn, candidates):
        return Hypothesis(self.tokens + [token], self.log_probs + [new_log_prob], new_state,
                          self.attns + [attn], self.candidates + [candidates])

    def __str__(self):
        return ('Hypothesis(log prob = %.4f, tokens = %s)' % (self.log_prob, self.tokens))

    def __repr__(self):
        return ('Hypothesis(log prob = %.4f, tokens = %s)' % (self.log_prob, self.tokens))


class BeamSearch:
    def __init__(self, decoder, encoder_outputs, decoder_hidden, output_lang,
                 beam_size=3, attentionOverrideMap=None, correctionMap=None, max_length=MAX_LENGTH):
        self.decoder = decoder
        self.encoder_outputs = encoder_outputs
        self.decoder_hidden = decoder_hidden
        self.beam_size = beam_size
        self.max_length = MAX_LENGTH
        self.output_lang = output_lang
        self.attention_override_map = attentionOverrideMap
        self.correction_map = correctionMap

        print("Attention Override:\n {}".format(attentionOverrideMap))
        print("Correction Map:\n {}".format(correctionMap))

    def decode_topk(self, latest_tokens, states, partials):

        # len(latest_tokens) x self.beam_size)
        topk_ids = [[0 for _ in range(self.beam_size)] for _ in range(len(latest_tokens))]
        topk_log_probs = [[0 for _ in range(self.beam_size)] for _ in range(len(latest_tokens))]
        new_states = [None for _ in range(len(states))]
        attns = [None for _ in range(len(states))]

        for token, state, i in zip(latest_tokens, states, range(len(latest_tokens))):
            decoder_input = Variable(torch.LongTensor([token]), volatile=True)

            attention_override = None
            if self.attention_override_map:
                if partials[i] in self.attention_override_map:
                    attention_override = self.attention_override_map[partials[i]]

            print("Hidden {}".format(state.data.numpy().tolist()[0][0][:3]))
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input,
                                                                             state,
                                                                             self.encoder_outputs,
                                                                             attention_override)

            if self.correction_map and partials[i] in self.correction_map:
                print("Corrected {} for partial= {}".format(self.correction_map[partials[i]], partials[i]))
                idx = self.output_lang.word2index[self.correction_map[partials[i]]]
                decoder_output.data[0][idx] = 100

            topk_v, topk_i = decoder_output.data.topk(self.beam_size)
            topk_v, topk_i = topk_v.numpy()[0], topk_i.numpy()[0]

            topk_ids[i] = topk_i.tolist()
            print("Token {}".format(self.output_lang.index2word[token]))
            print("Top: {}".format([self.output_lang.index2word[id] for id in topk_ids[i]]))

            decoder_output = nn.functional.log_softmax(decoder_output)
            topk_v, topk_i = decoder_output.data.topk(self.beam_size)
            topk_v, topk_i = topk_v.numpy()[0], topk_i.numpy()[0]

            topk_log_probs[i] = topk_v.tolist()
            print("Top: {}".format(topk_log_probs[i]))

            new_states[i] = decoder_hidden.clone()
            attns[i] = decoder_attention.data.numpy().tolist()[0]

        return topk_ids, topk_log_probs, new_states, attns

    def to_partial(self, tokens):
        return " ".join([self.output_lang.index2word[token] for token in tokens])

    def search(self):

        hyps = [Hypothesis([SOS_token], [0.0], self.decoder_hidden.clone(), None, [[]]) for _ in range(self.beam_size)]
        result = []

        steps = 0

        while steps < self.max_length and len(result) < self.beam_size:
            latest_tokens = [hyp.latest_token for hyp in hyps]
            states = [hyp.state for hyp in hyps]
            partials = [self.to_partial(hyp.tokens) for hyp in hyps]
            all_hyps = []

            num_beam_source = 1 if steps == 0 else len(hyps)
            topk_ids, topk_log_probs, new_states, attns = self.decode_topk(latest_tokens, states,
                                                                           partials)

            for i in range(num_beam_source):
                h, ns, attn = hyps[i], new_states[i], attns[i]

                for j in range(self.beam_size):
                    candidates = [self.output_lang.index2word[c] for c in (topk_ids[i][:j] + topk_ids[i][j + 1:])]
                    print(candidates)
                    all_hyps.append(
                        h.extend(topk_ids[i][j], topk_log_probs[i][j], ns, attn, candidates))

            # Filter
            hyps = []

            for h in self._best_hyps(all_hyps):
                if h.latest_token == EOS_token:
                    result.append(h)
                else:
                    hyps.append(h)
                if len(hyps) == self.beam_size or len(result) == self.beam_size:
                    break
            print(
                [[(self.output_lang.index2word[token], token, h.log_probs[i]) for i, token in enumerate(h.tokens)] for h
                 in
                 hyps])
            steps += 1

        return self._best_hyps(result, normalize=True)

    def _best_hyps(self, hyps, normalize=False):
        """Sort the hyps based on log probs and length.
        Args:
          hyps: A list of hypothesis.
        Returns:
          hyps: A list of sorted hypothesis in reverse log_prob order.
        """
        if normalize:
            return sorted(hyps, key=lambda h: h.log_prob / len(h.tokens), reverse=True)
        else:
            return sorted(hyps, key=lambda h: h.log_probs[-1], reverse=True)
