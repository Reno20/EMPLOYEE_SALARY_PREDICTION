import streamlit as st
import pandas as pd
import joblib

#  Page Configuration 
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="wide" 
)

#  Model Loading 
@st.cache_resource
def load_model():
    """Load the trained pipeline from disk."""
    try:
        model = joblib.load('best_salary_predictor.pkl')
        return model
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error(" Model file not found! Please ensure 'best_salary_predictor.pkl' is in the same directory.")
    st.stop()


st.title("Employee Salary Prediction App")
st.markdown("Predict whether an employee's income is >$50K or <=$50K based on their profile.")
st.markdown("---")


# Sidebar for User Inputs 
st.sidebar.header("Input Employee Details")

with st.sidebar:
    # Input fields for all the features the model was trained on
    age = st.slider("Age", 17, 90, 38)
    workclass = st.selectbox("Work Class", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov'])
    fnlwgt = st.number_input("Final Weight (fnlwgt)", 12285, 1484705, 189188)
    educational_num = st.slider("Education Level (Numeric)", 1, 16, 13)
    marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    occupation = st.selectbox("Occupation", ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces'])
    relationship = st.selectbox("Relationship", ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])
    race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    capital_gain = st.number_input("Capital Gain", 0, 99999, 0)
    capital_loss = st.number_input("Capital Loss", 0, 4356, 0)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Iran', 'Portugal', 'Nicaragua', 'Peru', 'Greece', 'France', 'Ecuador', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Cambodia', 'Laos', 'Thailand', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras', 'Scotland'])




input_data = pd.DataFrame({
    'age': [age], 'workclass': [workclass], 'fnlwgt': [fnlwgt],
    'educational-num': [educational_num], 'marital-status': [marital_status],
    'occupation': [occupation], 'relationship': [relationship], 'race': [race],
    'gender': [gender], 'capital-gain': [capital_gain], 'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week], 'native-country': [native_country]
})

st.subheader("Current Input Data")
st.dataframe(input_data)

#  Prediction Logic 
if st.button("Predict Salary Class", type="primary"):

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success(f"The predicted salary is **> $50K** with a confidence of **{prediction_proba[0][1]:.2%}**.")
    else:
        st.info(f"The predicted salary is **<= $50K** with a confidence of **{prediction_proba[0][0]:.2%}**.")


st.markdown("---")
st.subheader("Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction. The file must have the same columns as the input data.", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        batch_data = pd.read_csv(uploaded_file)
        
        
        required_columns = input_data.columns
        if not all(col in batch_data.columns for col in required_columns):
             st.error(f"Uploaded CSV is missing required columns. Please include: {', '.join(required_columns)}")
        else:
            # Make predictions
            batch_predictions = model.predict(batch_data[required_columns])
            batch_data['Predicted_Income'] = ['>$50K' if p == 1 else '<=$50K' for p in batch_predictions]

            st.write(" Predictions Complete")
            st.dataframe(batch_data)

            # Provide a download link for the results
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv(batch_data)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='salary_predictions.csv',
                mime='text/csv',
            )
    except Exception as e:
        st.error(f"An error occurred: {e}")
