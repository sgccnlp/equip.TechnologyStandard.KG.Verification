from typing import Tuple, List, Dict

import json
import numpy as np
from rouge import Rouge
from helper import rouge_score, snippet_projection, do_code_to_index, do_subtitle_format, argument_empty_process, generate_rouge_text
import copy
from itertools import permutations, combinations

FAULT_ID = "-1"


class IRMetric:
    """
    做检索评分的时候，有一个基本假设:
    对于同一个query给出的refs，任意两个refs不能具有相同的条款号(technology_standard_subtitle)。
    上述情况已经解决，现在可以评测相同条款号。

    评分分为三部分：mrr, recall, snippet。
    mrr：不受上述的问题（不同hyp具有同样的条款号）的影响
    recall：如果有两个hyp指向同一个ref，那么在计算的时候只能计算为1.因为只召回了一个ref文档。
    rouge：对rouge_matrix进行遍历，取最高得分的结果。此外，rouge是专注评价rouge这个指标的。所以只会在命中的文档上做平均。

    返回包括两个部分，评分和用于展示的lcs。

    
    """
    def __init__(self, ref_file, debug=False):
        self.debug = debug
        self.rouge = Rouge(exclusive=False)
        self.refs = {}
        with open(ref_file, encoding='utf8') as fp:
            for line in fp:
                inst = json.loads(line)
                doc_id = inst['id']
                self.refs[doc_id] = inst

    def get_scores(self, hyp):
        hyp = json.loads(hyp)
        refs = self.refs
        if not self.debug:
            try:
                scores, matched_snippets = self._get_scores(hyp, refs)
                if matched_snippets == {}:
                    matched_snippets = []
                return {"score": scores, "matched_snippets": matched_snippets}
            except:
                scores = {"recall": 0, "mrr": 0, "rouge": 0, "total": 0}
                return {"score": scores, "matched_snippets": []}
        else:
            scores, matched_snippets = self._get_scores(hyp, refs)
            if matched_snippets == {}:
                matched_snippets = []
            return {"score": scores, "matched_snippets": matched_snippets}

    def _get_scores(self, hyp, refs):
        ref = refs.get(hyp["id"], None)
        if ref == None:
            return {"recall": 0, "mrr": 0, "rouge": 0, "total": 0}, []
        recall, mrr, rouge, projection = self._get_query_score(
            hyp, ref)
        total = 0.5 * recall + 0.35 * mrr + 0.15 * rouge  # 权重设置。
        matched_snippets = self.match_snippets(projection, hyp, ref)
        return {"recall": recall, "mrr": mrr, "rouge": rouge, "total": total}, matched_snippets

    def _get_query_score(self, hyps, refs):
        """对一个query的hyp和ref进行评分, 同时返回projection用于计算snippet对照。
        """
        hyps = hyps['results']
        if hyps is None or not isinstance(hyps, list):
            hyps = []
        # 只取选手返回答案的前5个
        hyps = hyps[:5]
        refs = refs['docs']

        relevant_matrix = self._get_relevant_matrix(hyps, refs)
        rouge_matrix = self._get_rouge_matrix(hyps, refs, relevant_matrix)
        mrr = self._get_mrr(relevant_matrix)
        recall, projection_ = self._get_recall(relevant_matrix)
        rouge, projection = self._get_rouge(rouge_matrix, projection_)
        return recall, mrr, rouge, projection

    @staticmethod
    def _is_relevant(hyp, ref):
        hyp_code = str(do_code_to_index(hyp['technology_standard_code']))
        ref_code = str(do_code_to_index(ref['technology_standard_code']))
        hyp_subtitle = do_subtitle_format(hyp["technology_standard_subtitle"])
        ref_subtitle = do_subtitle_format(ref["technology_standard_subtitle"])
        if hyp_code == ref_code and hyp_subtitle == ref_subtitle:
            return 1
        else:
            return 0

    def _get_relevant_matrix(self, hyps, refs):
        n_hyps = len(hyps)
        n_refs = len(refs)
        relevant_matrix = np.zeros((
            n_hyps,
            n_refs,
        ))
        for i, hyp in enumerate(hyps):
            for j, ref in enumerate(refs):
                relevant_matrix[i, j] = self._is_relevant(hyp, ref)
        return relevant_matrix

    def _get_rouge_matrix(self, hyps, refs, relevant_matrix):
        n_hyps = len(hyps)
        n_refs = len(refs)
        rouge_matrix = np.zeros((
            n_hyps,
            n_refs,
        ))
        for i, hyp in enumerate(hyps):
            for j, ref in enumerate(refs):
                hyp_snippet = generate_rouge_text(hyp["snippet"])
                ref_snippet = generate_rouge_text(ref["snippet"])
                rouge_matrix[i, j] = rouge_score(hyp_snippet, ref_snippet)
        rouge_matrix = rouge_matrix * relevant_matrix
        return rouge_matrix

    @staticmethod
    def _get_rouge(rouge_matrix, projections) -> Tuple[float, List[Tuple[int, int]]]:
        """从所有组合projections中选择最大的rouge组合，同时根据最大rouge的情况返回对应的hyp和ref的对应列表。

        """
        max_score = 0
        max_projection = []
        for projection in projections:
            score = 0
            for hyp_to_ref in projection:
                hyp_ix, ref_ix = hyp_to_ref[0], hyp_to_ref[1]
                score += rouge_matrix[hyp_ix, ref_ix]
            score = 0 if len(projection) == 0 else score / len(projection)
            if score > max_score:
                max_score = score
                max_projection = projection
        return max_score, max_projection

    @staticmethod
    def _get_mrr(relevant_matrix):
        for i in range(relevant_matrix.shape[0]):
            if relevant_matrix[i].sum() > 0:
                return 1 / (i + 1)
        return 0

    @staticmethod
    def _get_recall(relevant_matrix) -> Tuple[float, List]:
        """
        用全排列的方法来做
        注意ref的候选答案大于5个的时候，只取5。
        """
        n_hyps = relevant_matrix.shape[0]
        n_refs = relevant_matrix.shape[1]
        n = min(n_hyps, n_refs)  # n_hyp是小于等于5的
        projections = []
        scores = []
        max_score = 0
        ix = 0
        for ref_comb in combinations(list(range(n_refs)), n):
            for ref_perm in permutations(ref_comb):
                for hyp_comb in combinations(list(range(n_hyps)), n):
                    score = 0
                    projection = []
                    # 上面的操作已经保证了hyp_perm和ref_perm的元素都小于或等于5
                    for hyp_ix, ref_ix in zip(hyp_comb, ref_perm):
                        tmp_score = relevant_matrix[hyp_ix, ref_ix]
                        if tmp_score > 0:
                            projection.append((hyp_ix, ref_ix))
                            score += 1
                    max_score = max(score, max_score)
                    if score >= max_score:
                        projections.append(projection)
                        scores.append(score)
        if max_score == 0:
            return 0, []
        projections = [projection for score, projection in zip(scores, projections) if score == max_score]
        score = max_score / min(n_refs, 5)
        return score, projections

    def match_snippets(self, projections: List[Tuple[int, int]], hyp: Dict, ref: Dict) -> List[Dict]:
        """[summary]

        Args:
            projection (Tuple[Tuple[int, int]]): [description]
            hyp (Dict): [description]
            ref (Dict): [description]
        Returns:
            [
                {
                    "hyp_rank": 1,
                    "ref_rank": 2,
                    "hyp_text": List[str],
                    "hyp_lcs_indices": List[int],
                    "ref_text": List[str],
                    "hyp_lcs_indices": List[int]
                }
            ]
        """
        results = []
        if hyp["results"] is None or not isinstance(hyp["results"], list):
            hyp["results"] = []
        hyp_snippets = [e["snippet"] for e in hyp["results"]]
        ref_snippets = [e["snippet"] for e in ref["docs"]]
        for projection in projections:
            hyp_ix, ref_ix = projection
            hyp_snippet, ref_snippet = hyp_snippets[hyp_ix], ref_snippets[ref_ix]
            # 加入处理空字符串的情况
            hyp_snippet = argument_empty_process(hyp_snippet)
            ref_snippet = argument_empty_process(ref_snippet)
            _, hyp_lcs_indices, ref_lcs_indices = snippet_projection(hyp_snippet, ref_snippet)
            results.append({
                "hyp_rank": hyp_ix,
                "ref_rank": ref_ix,
                "hyp_text": hyp_snippet,
                "hyp_lcs_indices": hyp_lcs_indices,
                "ref_text": ref_snippet,
                "ref_lcs_indices": ref_lcs_indices
            })
        return results


if __name__ == "__main__":
    ref_file = "data/ir/refs.json"
    hyp_file = "data/ir/hyps.json"
    metric = IRMetric(ref_file)
    with open(hyp_file) as fp:
        for line in fp:
            scores = metric.get_scores(line)
            print(scores['score'])
