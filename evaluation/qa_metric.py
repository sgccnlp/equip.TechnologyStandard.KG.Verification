"""
只对单个文件进行处理，可以默认为json字符串

既然多步推理的答案的条款号可能不止一个，那么标准名称/标准号是否也可能不止一个？
即答案来自多个标准。
目前写的评测是基于这样一个假定：一个答案只来自同一个标准，但是可能来自这个标准下的多个条款。发布时，该句应予以删除，否则泄漏信息。
上述假设既不成立也没必要：
不成立，因为同一个问题可能对应了多个答案形式。在目前的记录中，这些答案形式是具有相同的subtitle等。
没必要，因为现在改为输入hyp进行查询，而不是原本的多个问题的hyp同时进行评测。只需要一个hyp对所有ref进行评测就行，而不用考虑hyp之间冲突的情况。（例如某两个hyp的问题id相同了等等）
"""
from typing import Tuple, List, Dict

from helper import rouge_score, snippet_projection, do_subtitle_format, do_code_to_index, argument_empty_process, generate_rouge_text
import json

FAULT_ID = "-1"

class QAMetric:
    def __init__(self, ref_file, debug=False):
        self.answer_func = {
            '数字类': self._num_score,
            '判断类': self._judge_score,
            '抽取类': self._extr_score,
            '统计类': self._stat_score
        }
        self.debug = debug
        self.refs = {}
        with open(ref_file, encoding='utf8') as fp:
            for line in fp:
                ref = json.loads(line)
                ref_id = ref['id']
                self.refs[ref_id] = ref

    def get_scores(self, hyp):
        hyp = json.loads(hyp)
        refs = self.refs
        score = self._get_scores(hyp, refs)
        if score["matched_snippet"] == {}:
            score["matched_snippet"] = []
        return score

    def _get_scores(self, hyp, refs) -> Dict[str, float]:
        """[summary]

        Args:
            hyp ([type]): [description]
            refs ([type]): [description]

        Returns:
            Dict: [description]
            mode: int: 表示是否需要进行比较。即在没有匹配的id、数字类型、完全答错的情况下，是不需要比较的。
                0 表示可以比较，(参考答案不是数字类，且答对了)
                1 表示是参考答案是数字类, 所以不用比较，
                2 表示参考答案不是数字类，且回答答错了，无法比
                3 表示没有和hyp匹配的ref id
                4 表示出现了没有捕获的异常
        """
        if self.debug:
            if hyp["id"] not in refs.keys():
                return {"score": 0, "mode": 3, "matched_snippet": []}
            else:
                _id = hyp["id"]
                ref = refs[hyp["id"]]
                hyp = hyp["results"][0]
                score, mode, matched_snippet = self._get_q_score(hyp, ref)
                return {"score": score, "mode": mode, "matched_snippet": matched_snippet}
        else:
            try:
                if hyp["id"] not in refs.keys():
                    return {"score": 0, "mode": 3, "matched_snippet": []}
                else:
                    _id = hyp["id"]
                    ref = refs[hyp["id"]]
                    hyp = hyp["results"][0]
                    score, mode, matched_snippet = self._get_q_score(hyp, ref)
                    return {"score": score, "mode": mode, "matched_snippet": matched_snippet}
            except:
                return {"score": 0, "mode": 4, "matched_snippet": []}

    def _get_q_score(self, hyp, ref) -> Tuple[float, int, List[Dict]]:
        """返回最大分数和对应的判据的对比

        Args:
            hyp ([type]): [description]
            ref ([type]): [description]

        Returns:
            Tuple[float, Dict]: Dict是比对。
                {
                    "ref_ix": 1,
                    "ref_text": "",
                    "ref_lcs_common": [1, 2, 3],
                    "hyp_text": "",
                    "hyp_lcs_common": [1, 2, 3]
                }
            int: 表示是否需要进行比较。即在数字和完全答错的情况下，是不需要比较的。
                0表示可以比较，(参考答案不是数字类，且答对了)
                1表示是参考答案是数字类, 所以不用比较，
                2表示参考答案不是数字类，且回答答错了，无法比较，以及选手答案中的answer_type不属于预设的4个类型.
        """
        if hyp["answer_type"] not in self.answer_func.keys():
            return 0, 2, []
        if ref["answer_type"] != hyp['answer_type']:
            if ref["answer_type"] != "数字类":
                return 0, 2, []
            else:
                return 0, 1, []
        else:
            max_score = 0
            max_ix = -1
            for ref_ix, r in enumerate(ref['references']):
                score = self._get_one_r_score(hyp, r)
                if score >= max_score:
                    max_score = score
                    max_ix = ref_ix
            if ref["answer_type"] == "数字类":
                return max_score, 1, []
            else:
                if max_ix >= 0:
                    hyp_argument = hyp["answer"] if hyp["answer_type"] == "抽取类" else hyp["answer"]["argument"]
                    ref_argument = ref["references"][ref_ix]["answer"]["argument"]  # 在ref中，保持了其他抽取类、统计类、判断类的一致性
                    hyp_argument = argument_empty_process(hyp_argument)
                    ref_argument = argument_empty_process(ref_argument)
                    _, hyp_lcs_indices, ref_lcs_indices = snippet_projection(list(hyp_argument), list(ref_argument))
                    return max_score, 0, [{
                        "ref_ix": ref_ix,
                        "ref_text": ref_argument,
                        "ref_lcs_indices": ref_lcs_indices,
                        "hyp_text": hyp_argument,
                        "hyp_lcs_indices": hyp_lcs_indices
                    }]
                else:
                    return max_score, 2, []

    def _get_one_r_score(self, hyp, ref):
        """[summary]

        Args:
            hyp ([type]): [description]
            ref ([type]): [description]

        Returns:
            [float]: 当返回值为-1的时候，说明hyp和r不匹配。在匹配的情况下，score的取值范围是0，1的闭区间。
        """
        # TODO 同ie的_is_relevant
        hyp_code = str(do_code_to_index(hyp['technology_standard_code']))
        ref_code = str(do_code_to_index(ref['technology_standard_code']))
        hyp_subtitle = do_subtitle_format(hyp["technology_standard_subtitle"])
        ref_subtitle = do_subtitle_format(ref["technology_standard_subtitle"])
        if hyp_code == ref_code and hyp_subtitle == ref_subtitle:
            return self.answer_func[hyp['answer_type']](hyp, ref)
        else:
            return -1

    def _num_score(self, hyp, r):
        """数字类型答案的判断

        # TODO 数字部分的比较是否需要将数字进行转换。涉及到两个情况：
        1. 如果不进行运算，那么可以认为数字就是和文件中的一样。这样就可以直接进行字符串比较。
            但是这样有几个弊端：不同厂家对于数字的处理可能有偏差？例如末尾的0是否要带。
            如果单位要进行换算怎么办？
        2. 进行运算。
            如果文本中是：分数的情况呢？

        最简单的方法还是进行限制，给出题目的答案不考虑分数的情况。并且加入数值判断。例如万分之一等等。
        hyp_num 在 [ref_num * (1 - eps), ref_num * (1 + eps)]范围内就算对。
        """

        def process_none_num(num):
            none_num = "none"
            if num is None or num.strip() == "":
                return none_num
            else:
                return num

        def process_none_unit(unit):
            none_unit = "none"
            if unit is None or unit.strip() == "":
                return none_unit
            else:
                return unit

        h_answer = hyp['answer']
        h_num = process_none_num(h_answer["num"])
        h_unit = process_none_unit(h_answer["unit"])

        r_answer = r['answer']
        r_num = process_none_num(r_answer["num"])
        r_unit = process_none_unit(r_answer["unit"])

        return 1 if r_num == h_num and r_unit == h_unit else 0

    def _extr_score(self, hyp, r):
        h_argument = argument_empty_process(hyp["answer"])
        r_argument = argument_empty_process(r['answer']["argument"])
        r_kwargs = r["answer"]["kwargs"]

        kwarg_score = self.calculate_kwarg_score(h_argument, r_kwargs)
        
        h_answer = generate_rouge_text(h_argument)
        r_answer = generate_rouge_text(r_argument)
        return kwarg_score * rouge_score(h_answer, r_answer)

    def _stat_score(self, hyp, r):
        def process_none_num(num):
            none_num = "none"
            if num is None or num.strip() == "":
                return none_num
            else:
                return num

        h_answer= hyp['answer']
        h_num = process_none_num(h_answer["num"])
        h_argument = argument_empty_process(h_answer["argument"])

        r_answer = r['answer']
        r_num = process_none_num(r_answer["num"])
        r_argument = argument_empty_process(r_answer["argument"])

        r_kwargs = r["answer"]["kwargs"]
        kwarg_score = self.calculate_kwarg_score(h_argument, r_kwargs)

        # 统计量不同，得分为0
        if h_num != r_num:
            return 0
        else:
            h_argument = generate_rouge_text(h_argument)
            r_argument = generate_rouge_text(r_argument)
            return kwarg_score * rouge_score(h_argument, r_argument)

    def _judge_score(self, hyp, r):
        def process_judgement(judgement):
            if judgement != "是" and judgement != "否":
                return "none_judgement"
            else:
                return judgement

        h_answer= hyp['answer']
        h_tf = process_judgement(h_answer["judgement"])
        h_argument = argument_empty_process(h_answer["argument"])

        r_answer = r['answer']
        r_tf = process_judgement(r_answer["judgement"])
        r_argument = argument_empty_process(r_answer["argument"])
        r_kwargs = r["answer"]["kwargs"]
        kwarg_score = self.calculate_kwarg_score(h_argument, r_kwargs)

        if r_tf != h_tf:
            return 0
        h_argument = generate_rouge_text(h_argument)
        r_argument = generate_rouge_text(r_argument)
        return kwarg_score * rouge_score(h_argument, r_argument)

    @staticmethod
    def calculate_kwarg_score(hyp, kwargs: List[str]):
        kwargs = [x for x in kwargs if x != "" and x is not None]  # 先进行过滤
        n_kwargs = len(kwargs)
        n_right = 0
        if n_kwargs == 0:
            return 1
        for kwarg in kwargs:
            if kwarg in hyp:
                n_right += 1 
        return n_right / n_kwargs


if __name__ == "__main__":
    hyp_file = 'data/qa/hyps.json'
    ref_file = 'data/qa/refs.json'

    qa_metric = QAMetric(ref_file)
    with open(hyp_file) as fp:
        for line in fp:
            scores = qa_metric.get_scores(line)
            print(scores["score"])
