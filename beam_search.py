import torch
import torch.nn as nn
from torch.autograd import Variable
from hp import MAX_LENGTH, SOS_token, EOS_token, UNK_token
import math
from subword_nmt.apply_bpe import BPE
import hp

use_cuda = torch.cuda.is_available()


class Hypothesis:
    def __init__(self, tokens, words, log_probs, state, last_attn_vector, attns=None, candidates=None, is_unk=None):
        self.tokens = tokens
        self.words = words
        self.log_probs = log_probs
        self.state = state
        self.last_attn_vector = last_attn_vector
        self.attns = [[]] if attns is None else attns
        # candidate tokens at each search step
        self.candidates = candidates
        self.is_golden = False
        self.is_unk = is_unk
        self.alpha = 0.5
        self.beta = 0.5

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        return sum(self.log_probs)

    def extend(self, token, word, new_log_prob, new_state, last_attn_vector, attn, candidates, is_unk):
        h = Hypothesis(self.tokens + [token], self.words + [word], self.log_probs + [new_log_prob], new_state,
                       last_attn_vector,
                       self.attns + [attn], self.candidates + [candidates], self.is_unk + [is_unk])
        h.alpha = self.alpha
        h.beta = self.beta

        return h

    def __str__(self):
        return ('Hypothesis(log prob = %.4f, tokens = %s)' % (self.log_prob, self.tokens))

    def __repr__(self):
        return ('Hypothesis(log prob = %.4f, tokens = %s)' % (self.log_prob, self.tokens))

    def score(self):
        return self.log_prob / self.length_norm() + self.coverage_norm()

    def length_norm(self):
        return (5 + len(self.tokens)) ** self.alpha / (5 + 1) ** self.alpha

    def coverage_norm(self):
        # See http://opennmt.net/OpenNMT/translation/beam_search/

        # -1 for SOS token
        X = len(self.attns[0][0])
        Y = len(self.tokens) - 1

        res = 0
        for j in range(X):
            sum_ = 0
            for i in range(Y):
                sum_ += self.attns[i][0][j]
            res += math.log(min(1, sum_)) if sum_ > 0 else 0

        return self.beta * res


class BeamSearch:
    def __init__(self, decoder, encoder_outputs, decoder_hidden, output_lang,
                 beam_size=3, attentionOverrideMap=None, correctionMap=None, unk_map=None, beam_length=0.5,
                 beam_coverage=0.5, max_length=MAX_LENGTH):
        self.decoder = decoder
        self.encoder_outputs = encoder_outputs
        self.decoder_hidden = decoder_hidden
        self.beam_size = beam_size
        self.max_length = MAX_LENGTH
        self.output_lang = output_lang
        self.attention_override_map = attentionOverrideMap
        self.correction_map = correctionMap
        self.unk_map = unk_map
        self.beam_length = beam_length
        self.beam_coverage = beam_coverage
        self.bpe = BPE(open(hp.bpe_file))
        self.prefix = self.compute_prefix()
        self.process_corrections()

    def compute_prefix(self):
        if not self.correction_map:
            return
        print(self.correction_map)
        assert len(list(self.correction_map.keys())) == 1

        prefix = list(self.correction_map.keys())[0]
        correction = self.correction_map[prefix]

        prefix = prefix + " " + correction
        raw_words = prefix.split(" ")[1:]
        bpe_words = [self.bpe.process_line(word) if not word.endswith("@@") else word for word in raw_words]
        words = [word for bpe_word in bpe_words for word in bpe_word.split(" ")]
        return words

    def process_corrections(self):
        """Apply BPE to correction map, ignoring words that are already BPE'd"""
        if not self.correction_map:
            return
        prefixes = list(self.correction_map.keys())

        for prefix in prefixes:
            raw_words = self.correction_map.pop(prefix).split(" ")
            bpe_words = [self.bpe.process_line(word) if not word.endswith("@@") else word for word in raw_words]
            words = [word for bpe_word in bpe_words for word in bpe_word.split(" ")]

            for i in range(len(words)):
                curr_prefix = " ".join(prefix.split(" ") + words[:i])
                self.correction_map[curr_prefix] = words[i]

    def decode_topk(self, latest_tokens, states, last_attn_vectors, partials):
        """Decode all current hypotheses on the beam, returning len(hypotheses) x beam_size candidates"""

        # len(latest_tokens) x self.beam_size)
        topk_ids = [[0 for _ in range(self.beam_size)] for _ in range(len(latest_tokens))]
        topk_log_probs = [[0 for _ in range(self.beam_size)] for _ in range(len(latest_tokens))]
        new_states = [None for _ in range(len(states))]
        new_attn_vectors = [None for _ in range(len(states))]
        attns = [None for _ in range(len(states))]
        topk_words = [["" for _ in range(self.beam_size)] for _ in range(len(latest_tokens))]
        is_unk = [False for _ in range(len(latest_tokens))]

        # Loop over all hypotheses
        for token, state, attn_vector, i in zip(latest_tokens, states, last_attn_vectors, range(len(latest_tokens))):
            decoder_input = Variable(torch.LongTensor([token]))

            if use_cuda:
                decoder_input = decoder_input.cuda()

            attention_override = None
            if self.attention_override_map:
                if partials[i] in self.attention_override_map:
                    attention_override = self.attention_override_map[partials[i]]

            decoder_output, decoder_hidden, decoder_attention, last_attn_vector = self.decoder(decoder_input,
                                                                                               state,
                                                                                               self.encoder_outputs,
                                                                                               attn_vector,
                                                                                               attention_override)

            top_id = decoder_output.data.topk(1)[1]
            if use_cuda:
                top_id = top_id.cpu()

            top_id = top_id.numpy()[0].tolist()[0]

            if top_id == UNK_token:
                print("UNK found partial = {}".format(partials[i]))
            if top_id == UNK_token and self.unk_map and partials[i] in self.unk_map:
                # Replace UNK token based on user given mapping
                word = self.unk_map[partials[i]]
                topk_words[i][0] = word
                print("Replaced UNK token with {}".format(word))
                if word not in self.output_lang.word2index:
                    is_unk[i] = True
                else:
                    idx = self.output_lang.word2index[word]
                    decoder_output.data[0][idx] = 1000
            elif self.correction_map and partials[i] in self.correction_map:
                word = self.correction_map[partials[i]]
                print("Corrected {} for partial= {}".format(word, partials[i]))
                if not word in self.output_lang.word2index:
                    topk_words[i][0] = word
                    is_unk[i] = True
                idx = self.output_lang.word2index[word]
                decoder_output.data[0][idx] = 1000

            decoder_output = nn.functional.log_softmax(decoder_output)
            topk_v, topk_i = decoder_output.data.topk(self.beam_size)
            if use_cuda:
                topk_v, topk_i = topk_v.cpu(), topk_i.cpu()
            topk_v, topk_i = topk_v.numpy()[0], topk_i.numpy()[0]

            topk_ids[i] = topk_i.tolist()
            topk_log_probs[i] = topk_v.tolist()
            topk_words[i] = [self.output_lang.index2word[id] if not topk_words[i][j] else topk_words[i][j] for j, id in
                             enumerate(topk_ids[i])]

            new_states[i] = tuple(h.clone() for h in decoder_hidden)
            new_attn_vectors[i] = last_attn_vector.clone()
            attns[i] = decoder_attention.data
            if use_cuda:
                attns[i] = attns[i].cpu()
            attns[i] = attns[i].numpy().tolist()[0]

        return topk_ids, topk_words, topk_log_probs, new_states, new_attn_vectors, attns, is_unk

    def to_partial(self, tokens):
        return " ".join([self.output_lang.index2word[token] for token in tokens])

    def init_hypothesis(self):
        start_attn = []
        last_attn_vector = torch.zeros((1, self.decoder.hidden_size))

        if use_cuda:
            last_attn_vector = last_attn_vector.cuda()

        if not self.correction_map:
            return [Hypothesis([SOS_token], [self.output_lang.index2word[SOS_token]], [0.0],
                               tuple(h.clone() for h in self.decoder_hidden),
                               last_attn_vector.clone(),
                               start_attn, [[]], [False]) for _ in range(self.beam_size)]

        # Assume at most 1 correction prefix at all times
        prefix = [SOS_token] + [self.output_lang.word2index[token] for token in self.prefix]
        # We need: hidden state at the end of prefix, last_attn_vector, attention, tokens, candidates

        tokens = []
        candidates = []
        decoder_hidden = tuple(h.clone() for h in self.decoder_hidden)

        curr_candidates = []
        attn = [[0 for _ in range(self.encoder_outputs.size(0))]]

        for token in prefix:
            decoder_input = Variable(torch.LongTensor([token]))

            # Compute
            if use_cuda:
                decoder_input = decoder_input.cuda()
            decoder_output, decoder_hidden, decoder_attention, last_attn_vector = self.decoder(decoder_input,
                                                                                               decoder_hidden,
                                                                                               self.encoder_outputs,
                                                                                               last_attn_vector)

            # Update
            start_attn += [attn]
            tokens += [token]
            candidates += [curr_candidates]

            attn = decoder_attention.data
            if use_cuda:
                attn = attn.cpu()
            attn = attn.numpy().tolist()[0]

            topk_v, topk_i = decoder_output.data.topk(self.beam_size)
            if use_cuda:
                topk_v, topk_i = topk_v.cpu(), topk_i.cpu()
            topk_v, topk_i = topk_v.numpy()[0], topk_i.numpy()[0]

            topk_ids = topk_i.tolist()
            topk_log_probs = topk_v.tolist()
            curr_candidates = [self.output_lang.index2word[i] for i in topk_ids]

        return [Hypothesis(list(tokens), [self.output_lang.index2word[token] for token in tokens], [0.0] * len(tokens),
                           tuple(h.clone() for h in decoder_hidden),
                           last_attn_vector.clone(),
                           list(start_attn), candidates, [False] * len(tokens)) for _ in range(self.beam_size)]

    def search(self):

        start_attn = [[[0 for _ in range(self.encoder_outputs.size(0))]]]
        last_attn_vector = torch.zeros((1, self.decoder.hidden_size))

        if use_cuda:
            last_attn_vector = last_attn_vector.cuda()

        hyps = self.init_hypothesis()

        for h in hyps:
            h.alpha = self.beam_length
            h.beta = self.beam_coverage

        result = []

        steps = 0

        while steps < self.max_length * 2 and len(result) < self.beam_size:
            latest_tokens = [hyp.latest_token for hyp in hyps]
            states = [hyp.state for hyp in hyps]
            partials = [self.to_partial(hyp.tokens) for hyp in hyps]
            last_attn_vectors = [hyp.last_attn_vector for hyp in hyps]
            all_hyps = []

            num_beam_source = 1 if steps == 0 else len(hyps)
            topk_ids, topk_words, topk_log_probs, new_states, new_attn_vectors, attns, is_unk = self.decode_topk(
                latest_tokens, states, last_attn_vectors,
                partials)

            for i in range(num_beam_source):
                h, ns, av, attn = hyps[i], new_states[i], new_attn_vectors[i], attns[i]

                for j in range(self.beam_size):
                    candidates = [self.output_lang.index2word[c] for c in (topk_ids[i][:j] + topk_ids[i][j + 1:])]

                    all_hyps.append(
                        h.extend(topk_ids[i][j], topk_words[i][j], topk_log_probs[i][j], ns, av, attn, candidates,
                                 is_unk[i]))

            # Filter
            hyps = []
            # print("All Hyps")
            for h in all_hyps:
                pass
            # print([(word, log_prob) for word, log_prob in zip(h.words, h.log_probs)])
            # print("====")

            for h in self._best_hyps(all_hyps):
                if h.latest_token == EOS_token:
                    result.append(h)
                else:
                    hyps.append(h)
                if len(hyps) == self.beam_size or len(result) == self.beam_size:
                    break
            steps += 1

        # print("Beam Search found {} hypotheses for beam_size {}".format(len(result), self.beam_size))
        res = self._best_hyps(result, normalize=True)
        if res:
            res[0].is_golden = True
        return res

    def _best_hyps(self, hyps, normalize=False):
        """Sort the hyps based on log probs and length.
        Args:
          hyps: A list of hypothesis.
        Returns:
          hyps: A list of sorted hypothesis in reverse log_prob order.
        """
        if normalize:
            return sorted(hyps, key=lambda h: h.score(), reverse=True)
        else:
            return sorted(hyps, key=lambda h: h.log_prob, reverse=True)
