from flask import Flask, request
import json
from ie_metric import IEMetric
from qa_metric import QAMetric


ie_ref_file = "data/data/ie/refs.json"
qa_ref_file = "data/data/qa/ref.json"

ie_metric = IEMetric(ie_ref_file)
qa_metric = QAMetric(qa_ref_file)


app = Flask(__name__)


@app.route("/ie", methods=["POST"])
def ie():
    a = request.get_data()
    results = ie_metric.get_scores(a)
    return json.dumps(results, ensure_ascii=False)


@app.route("/qa", methods=["POST"])
def qa():
    a = request.get_data()
    results = qa_metric.get_scores(a)
    return json.dumps(results, ensure_ascii=False)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
