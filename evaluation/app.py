from flask import Flask, request
import json
from ir_metric import IRMetric
from qa_metric import QAMetric


ir_ref_file = "data/ir/refs.json"
qa_ref_file = "data/qa/refs.json"

ir_metric = IRMetric(ir_ref_file)
qa_metric = QAMetric(qa_ref_file)


app = Flask(__name__)


@app.route("/ir", methods=["POST"])
def ir():
    a = request.get_data()
    results = ir_metric.get_scores(a)
    return json.dumps(results, ensure_ascii=False)


@app.route("/qa", methods=["POST"])
def qa():
    a = request.get_data()
    results = qa_metric.get_scores(a)
    return json.dumps(results, ensure_ascii=False)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
