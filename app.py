from flask import Flask, render_template, request, redirect, url_for
from Classification.prediction import Predict
import re

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        global answer
        food_name = request.form['food_name']
        food_group = request.form['food_group']
        # sanitization
        food_group = food_group.lower()
        food_name = food_name.lower()
        food_group = re.sub(r'[^\w\s]', '', food_group)
        food_name = re.sub(r'[^\w\s]', '', food_name)

        new_input_from_form_list = []
        new_input_from_form = food_group
        new_input_from_form_list.append(new_input_from_form)
        prediction = Predict()

        answer = ''
        if food_group == 'water' or food_name == 'water':
            answer += 'Keep taking enough water to stay hydrated'
        else:
            answer += prediction.predict_food_outcome(new_input_from_form_list, food_name)
        return redirect(url_for('food_predict'))
    else:
        return render_template('index.html', title='Home')


@app.route('/predict-food', methods=['GET', 'POST'])
def food_predict():
    global answer
    return render_template('show.html', title='Prediction', answer=answer)


if __name__ == '__main__':
    app.run(port=8889, debug=True)
