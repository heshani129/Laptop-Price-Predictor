from flask import Flask, render_template, request
import pickle
import numpy as np

# Setup Flask application
app = Flask(__name__)

# Load the trained model once
with open('model/predictor.pickle', 'rb') as file:
    model = pickle.load(file)

# Lists of categories used during training (must match model)
company_list = ['acer', 'apple', 'asus', 'dell',
                'hp', 'lenovo', 'msi', 'other', 'toshiba']
typename_list = ['2in1convertible', 'gaming',
                 'netbook', 'notebook', 'ultrabook', 'workstation']
opsys_list = ['linux', 'mac', 'other', 'windows']
cpu_list = ['amd', 'intelcorei3', 'intelcorei5', 'intelcorei7', 'other']
gpu_list = ['amd', 'intel', 'nvidia']

# Function to create the feature vector


def make_feature_list(ram, weight, company, typename, opsys, cpu, gpu, touchscreen, ips):
    features = []

    # Numeric features
    features.append(int(ram))
    features.append(float(weight))

    # Boolean features (1 if checked, 0 if not)
    features.append(1 if touchscreen else 0)
    features.append(1 if ips else 0)

    # One-hot encoding for categorical features
    def encode_category(categories, value):
        return [1 if value == item else 0 for item in categories]

    features += encode_category(company_list, company)
    features += encode_category(typename_list, typename)
    features += encode_category(opsys_list, opsys)
    features += encode_category(cpu_list, cpu)
    features += encode_category(gpu_list, gpu)

    # Check feature length
    expected_length = model.n_features_in_
    if len(features) < expected_length:
        # Add zeros for missing categories (safe fallback)
        features += [0] * (expected_length - len(features))

    return features


@app.route('/', methods=['GET', 'POST'])
def index():
    pred_value = None

    if request.method == 'POST':
        # Get form inputs
        ram = request.form['ram']
        weight = request.form['weight']
        company = request.form['company']
        typename = request.form['typename']
        opsys = request.form['opsys']
        cpu = request.form['cpuname']
        gpu = request.form['gpuname']
        # checkbox returns 'on' if checked
        touchscreen = request.form.get('touchscreen')
        ips = request.form.get('ips')  # checkbox returns 'on' if checked

        touchscreen = True if touchscreen == 'on' else False
        ips = True if ips == 'on' else False

        # Prepare features
        features = make_feature_list(
            ram, weight, company, typename, opsys, cpu, gpu, touchscreen, ips)

        # Make prediction
        pred_value = model.predict([features])[0]
        # Adjust multiplier if needed
        pred_value = np.round(pred_value, 2) * 221

    return render_template('index.html', pred_value=pred_value)


if __name__ == '__main__':
    app.run(debug=True)
