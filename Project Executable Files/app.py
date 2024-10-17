from markupsafe import escape
from flask import Flask, request, render_template,redirect,url_for
import pickle
import numpy as np


app = Flask(__name__)
model = pickle.load(open(r'rfmodel.pkl', 'rb'))
scaler = pickle.load(open(r'scaler.pkl', 'rb'))
# Define mapping dictionaries
education_mapping = {"Graduate": 0, "Not Graduate": 1}
employed_mapping = {"Yes": 1, "No": 0}


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Fetch input values from the form
            dependents = int(request.form['dependents'])
            education = request.form['education']
            employed = request.form['employed']
            income_annum = int(request.form['income_annum'])
            LoanAmount = int(request.form['LoanAmount'])
            Loan_Term = int(request.form['Loan_Term'])
            cibil = int(request.form['cibil'])
            assets = int(request.form['assets'])

            print(f"Dependents: {dependents}, Education: {education}, Employed: {employed}, "
                  f"Income Annum: {income_annum}, Loan Amount: {LoanAmount}, Loan Term: {Loan_Term}, "
                  f"CIBIL: {cibil}, Assets: {assets}")

            # Use predefined mappings for education and employment status
            grad_s = education_mapping.get(education, 1)
            emp_s = employed_mapping.get(employed, 0)

            # Prepare input data for the model
            data = [[dependents, grad_s, emp_s, income_annum, LoanAmount, Loan_Term, cibil, assets]]
            data = scaler.transform(data)  # Apply scaling

            # Make prediction
            prediction = model.predict(data)

            if prediction[0] == 1:
                return redirect(url_for('loan_approved'))
            else:
                return redirect(url_for('loan_rejected'))

        except KeyError as e:
            return f"KeyError: {str(e)}. Please check your form data.", 400
        except ValueError as e:
            return f"ValueError: {str(e)}. Please check your input values.", 400
        except Exception as e:
            return str(e), 400

    return render_template('prediction.html')


@app.route('/loan_approved')
def loan_approved():
    return render_template("approved.html")  # Create this template for approved loans

@app.route('/loan_rejected')
def loan_rejected():
    return render_template('rejected.html')  # Create this template for rejected loans

if __name__ == "__main__":
    app.run(debug=True)