from flask import Flask, render_template, request
from predict import predict_loan

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        loan_amnt = float(request.form['loan_amnt'])
        int_rate = float(request.form['int_rate'])
        annual_inc = float(request.form['annual_inc'])
        grade = request.form['grade']
        dti = float(request.form['dti'])
        term = int(request.form['term'])

        # 🔒 Validation
        if loan_amnt <= 0 or loan_amnt > 50000:
            raise ValueError("Loan amount must be between 1 and 50,000")

        if annual_inc < 1000:
            raise ValueError("Income too low")

        if int_rate < 0 or int_rate > 50:
            raise ValueError("Interest rate must be between 0–50")

        if dti < 0 or dti > 100:
            raise ValueError("DTI must be between 0–100")

        if term not in [36, 60]:
            raise ValueError("Invalid loan term")

        data = {
            "loan_amnt": loan_amnt,
            "int_rate": int_rate,
            "annual_inc": annual_inc,
            "grade": grade,
            "dti": dti,
            "term": term
        }

        probability, prediction = predict_loan(data)

        # 🎯 Risk level
        if probability < 0.3:
            risk = "Low"
        elif probability < 0.5:
            risk = "Medium"
        else:
            risk = "High"

        return render_template(
            'index.html',
            prediction=prediction,
            probability=probability,
            risk=risk
        )

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)