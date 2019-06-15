from flask import Flask, request, render_template
from main import compute


app = Flask(__name__, instance_relative_config=True)


@app.route('/hello')
def hello():
    return 'Hello, World!'


@app.route('/nlp/<topic>', methods=['GET'])
def nlp(topic):
    print(topic)
    print('lol')
    result = ''
    if topic:
        result = compute(topic)

    print(result)
    return result


app.run(host='192.168.31.171')
