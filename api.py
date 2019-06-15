from flask import Flask, request, render_template
from main import compute


app = Flask(__name__, instance_relative_config=True)


@app.route('/nlp/<topic>', methods=['GET'])
def nlp(topic):
    result = ''
    if topic:
        result = compute(topic)
    return result


app.run(host='192.168.31.171')
