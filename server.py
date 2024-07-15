from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/data')
def get_data():
    # Example data loading
    data = pd.read_csv('data/temperature.csv', delimiter=',')
    return jsonify(data.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
