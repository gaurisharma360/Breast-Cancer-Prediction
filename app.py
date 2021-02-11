import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
 
app = Flask(__name__,static_url_path='/static')

# load model
model = load_model('model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    final_features = scaler.transform(final_features)    
    prediction = model.predict(final_features)
    y_probabilities_test = model.predict_proba(final_features)
    y_prob_success = y_probabilities_test[:, ]
    print("final features",final_features)
    print("prediction:",prediction)
    output = prediction[0]
    y_prob=y_prob_success[0]
    print(output)

    if output == "True":
        return render_template('index.html', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A BENIGN CANCER WITH PROBABILITY VALUE  {}'.format(y_prob))
    else:
         return render_template('index.html', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A MALIGNANT CANCER WITH PROBABILITY VALUE  {}'.format(y_prob))
        
@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)