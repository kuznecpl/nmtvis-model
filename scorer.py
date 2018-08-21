import math


class Scorer:
    def compute_scores(self, source, translation, attention, keyphrases):
        return {"coverage_penalty": self.coverage_penalty(attention),
                "coverage_deviation_penalty": self.coverage_deviation_penalty(attention),
                "confidence": self.confidence(attention),
                "length": len(source.split(" ")),
                "ap_in": self.absentmindedness_penalty_in(attention),
                "ap_out": self.absentmindedness_penalty_out(attention),
                "keyphrase_score": self.keyphrase_score(source, keyphrases, attention)
                }

    def keyphrase_score(self, sentence, keyphrases, attention):
        score = 0

        for word in sentence.replace("@@ ", "").split(" "):
            for keyphrase, freq in keyphrases:
                score += word.lower().count(keyphrase.lower()) * math.log(freq)
        return score

    def length_deviation(self, source, translation):
        source = source.split(" ")
        translation = translation.split(" ")

        X, Y = len(source), len(translation)

        return math.fabs(X - Y) / X

    def coverage_penalty(self, attention, beta=1):
        m, n = len(attention), len(attention[0])

        res = 0
        for j in range(n):
            sum_ = 0
            for i in range(m):
                sum_ += attention[i][j]
            res += math.log(min(1, sum_)) if sum_ > 0 else 0

        return -(1 / n) * beta * res

    def coverage_deviation_penalty(self, attention):
        m, n = len(attention), len(attention[0])

        res = 0
        for j in range(n):
            res += math.log(1 + (1 - sum([attention[i][j] for i in range(m)])) ** 2)
        return (1 / n) * res

    def absentmindedness_penalty_out(self, attention):
        m, n = len(attention), len(attention[0])

        sum_ = 0
        for row in attention:
            norm = sum(row)
            if norm > 0:
                normRow = [i / norm for i in row]
                sum_ += sum([(i * math.log(i) if i else 0) for i in normRow])

        return - (1 / m) * sum_

    def absentmindedness_penalty_in(self, attention):
        return self.absentmindedness_penalty_out(list(zip(*attention)))

    def confidence(self, attention):
        x = self.coverage_deviation_penalty(attention) + self.absentmindedness_penalty_in(
            attention) + self.absentmindedness_penalty_out(attention)

        return math.exp(-0.05 * (x ** 2))

    def length_penalty(self, attention):
        return len(attention[0])
