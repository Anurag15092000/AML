
from flask import Flask, request, render_template, url_for, redirect
import joblib
from score import score

app = Flask(__name__)
best_model= joblib.load('model_pipeline.pkl')
threshold = 0.5

@app.route('/') 
def home():
    return render_template('spam_page.html')


@app.route('/spam', methods=['POST'])
def spam():
    txt = request.form['sent']
    pred,prop = score(txt, best_model, threshold)
    label = "Spam" if pred == 1 else "Not spam"
    ans = f"""The sentence "{txt}" is {label} with propensity {prop}."""
    return render_template('result_page.html', ans = ans)


if __name__ == '__main__': 
    app.run(debug=True,use_reloader=False)