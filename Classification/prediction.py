from sklearn.externals import joblib
from Classification.training import Training
import sqlite3


class Predict(object):

    @staticmethod
    def predict_food_outcome(input_from_form_group, input_from_form):
        try:
            # import Training
            train_import = Training()
            # Load Dataset
            train_import.load_dataset()
            # Encode variables
            encoded_x = train_import.encoding_variables_test(input_from_form_group)
            train_import.split_to_train_and_test()
            x_test_scaled = train_import.feature_scaling_test(encoded_x)
            saved_model = joblib.load('./Classification/saved_model/food_prediction')
            encoded_answer = saved_model.predict(x_test_scaled)
            decoded_answer = train_import.decoding_y(encoded_answer)
            if decoded_answer == 0:
                return '{} is not recommended for ulcer patients, consult your doctor'.format(input_from_form)
            else:
                return '{} is recommended for ulcer patients, consider adding it to your daily meal!'.format(input_from_form)
        except Exception as _:
            connection = sqlite3.connect('./database/Food_Logs.db')
            cursor = connection.cursor()
            query = "INSERT INTO food_logs VALUES (NULL, ?, ?)"
            cursor.execute(query, (input_from_form, input_from_form_group[0],))
            connection.commit()
            connection.close()
            return "We have added this food to our interest list, our dietitians will take a look at it!"


if __name__ == '__main__':
    new_input_from_form_list = []
    new_input = 'Pawpaw'
    new_input_from_form = 'fruit'
    new_input_from_form_list.append(new_input_from_form)
    prediction = Predict()
    ret = prediction.predict_food_outcome(new_input_from_form_list, new_input)
    print(ret)
