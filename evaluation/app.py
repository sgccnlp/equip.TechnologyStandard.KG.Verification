from flask import Flask, request
import json
from ir_metric import IRMetric
from qa_metric import QAMetric
from pathlib import Path


profile_dir = "./tests/benchmark/"
benchmark = True  # 生产时需要设为False


ir_ref_file = "data/ir/refs.json"
qa_ref_file = "data/qa/refs.json"

ir_metric = IRMetric(ir_ref_file)
qa_metric = QAMetric(qa_ref_file)


app = Flask(__name__)
if benchmark:
    from werkzeug.middleware.profiler import ProfilerMiddleware
    profile_dir = Path(profile_dir)
    if not profile_dir.is_dir():
        profile_dir.mkdir(parents=True)
    app.wsgi_app = ProfilerMiddleware(app.wsgi_app, profile_dir=str(profile_dir))


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
