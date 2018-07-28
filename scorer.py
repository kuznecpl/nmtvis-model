import math


class Scorer:
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
        for i in range(m):
            for j in range(n):
                sum_ += attention[i][j] * math.log(attention[i][j]) if attention[i][j] > 0 else 0

        return - (1 / m) * sum_

    def absentmindedness_penalty_in(self, attention):
        m, n = len(attention), len(attention[0])

        sum_ = 0
        for i in range(m):
            for j in range(n):
                attn_sum = sum([attention[k][j] for k in range(m)])
                if attn_sum > 0:
                    attention[i][j] /= attn_sum
                else:
                    attention[i][j] = 0

                sum_ += attention[i][j] * math.log(attention[i][j]) if attention[i][j] > 0 else 0

        return - (1 / m) * sum_

    def confidence(self, attention):
        x = self.coverage_deviation_penalty(attention) + self.absentmindedness_penalty_in(
            attention) + self.absentmindedness_penalty_out(attention)

        return math.exp(-0.05 * (x ** 2))

    def length_penalty(self, attention):
        return len(attention[0])
