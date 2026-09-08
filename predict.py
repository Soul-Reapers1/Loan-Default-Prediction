import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load("loan_pipeline.pkl")

# Default values
base_sample = {
    "loan_amnt": 10000,
    "annual_inc": 50000,
    "int_rate": 10.0,
    "term": 36,
    "grade": "C",
    "dti": 20.0
}

def predict_loan(data):
    sample = base_sample.copy()

    # Override with user input
    sample['loan_amnt'] = min(float(data['loan_amnt']), 40000)
    sample['annual_inc'] = max(float(data['annual_inc']), 1000)
    sample['int_rate'] = float(data['int_rate'])
    sample['term'] = int(data['term'])
    sample['grade'] = str(data['grade']).upper()
    sample['dti'] = float(data['dti'])

    # Derived feature (IMPORTANT: same as training)
    sample['loan_to_income'] = sample['loan_amnt'] / sample['annual_inc']

    # Convert to DataFrame
    df_input = pd.DataFrame([sample])

    # Prediction
    prob = pipeline.predict_proba(df_input)[:, 1][0]

    # Decision logic (realistic thresholds)
    if prob < 0.3:
        decision = "Accept"
    elif prob < 0.5:
        decision = "Waiting for review"
    else:
        decision = "Reject"

    return round(prob, 4), decision


# import pandas as pd
# import joblib

# # Load trained pipeline
# pipeline = joblib.load("loan_pipeline.pkl")

# # ---------------- LOAD & CLEAN DATA ---------------- #

# df = pd.read_csv("loan.csv")

# drop_cols = ["loan_status", "id", "member_id", "url"]
# leakage_cols = [
#     "recoveries","collection_recovery_fee","last_pymnt_amnt",
#     "total_rec_prncp","total_rec_int","out_prncp",
#     "out_prncp_inv","total_pymnt","total_pymnt_inv",
#     "total_rec_late_fee","last_paymnt_d",
#     "last_credit_pull_d","zip_code","title","emp_title"
# ]

# df = df.drop(columns=drop_cols + leakage_cols, errors="ignore")

# # Safe conversions
# if 'term' in df.columns:
#     df['term'] = df['term'].astype(str).str.extract('(\d+)')[0]
#     df['term'] = pd.to_numeric(df['term'], errors='coerce')

# if 'int_rate' in df.columns:
#     df['int_rate'] = df['int_rate'].astype(str).str.replace('%','')
#     df['int_rate'] = pd.to_numeric(df['int_rate'], errors='coerce')

# # Clean DTI
# if 'dti' in df.columns:
#     df['dti'] = df['dti'].replace(-1, None)
#     df['dti'] = df['dti'].clip(upper=100)

# # ---------------- BETTER BASELINE ---------------- #

# # Use moderate-risk users instead of global average
# base_df = df[df['grade'] == 'C']

# num_defaults = base_df.select_dtypes(include=['int64','float64']).median().to_dict()
# cat_defaults = base_df.select_dtypes(include=['object']).mode().iloc[0].to_dict()

# base_sample = {**num_defaults, **cat_defaults}

# # ---------------- PREDICTION FUNCTION ---------------- #

# def predict_default(sample):
#     df_input = pd.DataFrame([sample])

#     # Align columns with training
#     # df_input = df_input.reindex(columns=pipeline.feature_names_in_, fill_value=0)
#     df_input = pd.DataFrame([sample])

#     prob = pipeline.predict_proba(df_input)[:, 1][0]
#     if prob < 0.02:
#         decision = "Accept"
#     elif prob < 0.05:
#         decision = "Waiting for review"
#     else:
#         decision = "Reject"
#     return prob, decision

# # ---------------- MAIN LOOP ---------------- #

# while True:

#     sample = base_sample.copy()

#     loan_amnt = float(input("Enter loan amount: "))
#     if loan_amnt > 40000:
#         print("Loan amount exceeding the limit. Defaulting to 40000")
#         loan_amnt = 40000
#     annual_inc = float(input("Enter annual income: "))
#     if annual_inc < 1000:
#         print("Annual income too low. Defaulting to 1000")
#         annual_inc = 1000
#     int_rate = float(input("Enter interest rate: "))
#     term = int(input("Enter term (36 or 60): "))
#     grade = input("Enter grade (A-G): ").upper()
#     dti = float(input("Enter DTI (%): "))

#     # Input validation
#     if dti < 0 or dti > 100:
#         print("Invalid DTI. Enter between 0 and 100.\n")
#         continue

#     if grade not in list("ABCDEFG"):
#         print("Invalid grade. Defaulting to C.\n")
#         grade = "C"

#     # Override key features
#     sample['loan_amnt'] = loan_amnt
#     sample['annual_inc'] = max(annual_inc, 1)  # avoid division crash
#     sample['int_rate'] = int_rate
#     sample['term'] = term
#     sample['grade'] = grade
#     sample['dti'] = dti

#     # CRITICAL FEATURE
#     sample['loan_to_income'] = sample['loan_amnt'] / sample['annual_inc']

#     prob, decision = predict_default(sample)

#     print(f"\nDefault Probability: {prob:.4f}")
#     print(f"Decision: {decision}\n")

#     cont = input("Add another customer [y/n]: ").lower()
#     if cont == 'n':
#         break
#     elif cont != 'y':
#         print("Invalid choice, continuing...\n")

# print("Predicted probability: ", prob)
# print(pipeline.classes_)