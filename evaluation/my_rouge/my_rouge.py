from typing import List

from collections import Counter


class MyRouge:

    def __init__(self):
        pass

    def get_score(self, hyp, ref):
        hyp_1grams = self.get_ngrams(hyp, 1)
        hyp_2grams = self.get_ngrams(hyp, 2)

        ref_1grams = self.get_ngrams(ref, 1)
        ref_2grams = self.get_ngrams(ref, 2)

        interset_1grams = self.intersection_grams(hyp_1grams, ref_1grams)
        interset_2grams = self.intersection_grams(hyp_2grams, ref_2grams)

        rouge1_p, rouge1_r, rouge1_f = self.get_ngram_p_r_f(interset_1grams, hyp_1grams, ref_1grams)
        rouge2_p, rouge2_r, rouge2_f = self.get_ngram_p_r_f(interset_2grams, hyp_2grams, ref_2grams)

        rougel_p, rougel_r, rougel_f = self.get_rouge_l(hyp, ref)

        return {"rouge-1": rouge1_r, "rouge-2": rouge2_r, "rouge-l": rougel_f}


    def get_words(self, text) -> List[str]:
        assert isinstance(text, str), f"text should be string, but get {text} which type is {type(text)}"
        words = [x.strip() for x in text.split() if x.strip() != ""]
        return words

    def get_ngrams(self, text, n):
        words = self.get_words(text)
        ngrams = Counter()
        maximum_start_index = len(words) - n
        for i in range(maximum_start_index + 1):
            ngrams[tuple(words[i:i+n])] += 1
        return ngrams

    def intersection_grams(self, a_grams: Counter, b_grams: Counter):
        inter_grams = Counter()
        for key in a_grams.keys():
            if key in b_grams:
                inter_grams[key] = min(a_grams[key], b_grams[key])
        return inter_grams

    def get_ngram_p_r_f(self, int_grams: Counter, hyp_grams: Counter, ref_grams: Counter):
        n_ref = sum(ref_grams.values())
        n_hyp = sum(hyp_grams.values())
        n_int = sum(int_grams.values())
        p = 0 if n_hyp == 0 else n_int / n_hyp
        r = 0 if n_ref == 0 else n_int / n_ref
        f = 0 if p + r == 0 else 2 * p * r / (p + r)
        return p, r, f

    def get_rouge_l(self, hyp, ref):
        hyp_words = self.get_words(hyp)
        ref_words = self.get_words(ref)

        n_hyp = len(hyp_words)
        n_ref = len(ref_words)
        if n_hyp == 0 or n_ref == 0:
            return 0, 0, 0

        matrix = [[0 for _ in range(n_ref+1)] for _ in range(n_hyp+1)]

        for i in range(n_hyp):
            for j in range(n_ref):
                if hyp_words[i] == ref_words[j]:
                    matrix[i+1][j+1] = matrix[i][j]+1
                else:
                    matrix[i+1][j+1] = max(matrix[i+1][j], matrix[i][j+1])
        n_int = matrix[-1][-1]
        p = n_int / n_hyp if n_hyp != 0 else 0
        r = n_int / n_ref if n_ref != 0 else 0
        f = 2*p*r / (p+r) if p+r != 0 else 0
        return p, r, f
