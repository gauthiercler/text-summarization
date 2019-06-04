from flask import Flask, request, render_template
from main import Main


app = Flask(__name__, instance_relative_config=True)


@app.route('/hello')
def hello():
    return 'Hello, World!'


@app.route('/nlp')
def nlp():
    result = ""
    if "subject" in request.args:
        m = Main()
        m.run(request.args.get("subject"))
        result = m.sentences
    return render_template('nlp.html', summary=result)

