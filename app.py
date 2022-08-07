from re import L
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from joblib import load


#loading the dataset
test=pd.read_csv("input/test_lAUu6dG.csv") 
train=pd.read_csv("input/train_ctrUa4K.csv")

#container organization for the parts
# header=st.container()
# model_prediction=st.container()


#organizations
#header
#dataset introduction
    #where is the dataset from
    #Explanation of dataset
# Data Cleaning
# Feature Generation
# Model Training Explanation
    # models used
    # model accuracy and confusion matrix for each
    # tabulations and graphs
# Model Demo

st.title('Loan Approval Prediction using Machine Learning')

st.markdown("""

The dataset comes from a company that deals with housing loans. They have presence in urban, rural and semi urban areas.
Customer applies for a loan, which is then used by the bank to validate the customer's eligibility for a loan. 
We are now automating this process of approval of loans using Machine learning. We get details of their Gender, Education, 
Marital Status, Income, Loan amount and term required, among others.

We work on who are eligible for loans, so that the bank can save resources by targeting that demographic. 

Feel Free to explore the tabs below to see how I had approached this problem from the exploration of dataset 
to the model deployment in this website

---
## Model Demo

""")

# Demo of the model
col_a,col_b=st.columns(2)
Gender=col_a.selectbox('Gender of the applicant', ('Female', 'Male'))
Married=col_a.checkbox('Are you Married?')
Dependents=col_b.number_input('Number of Dependents?',step=1,format='%d')
Education=col_a.checkbox('Are you a College Graduate?')
Self_Employed=col_b.checkbox('Are you Self Employed?')
ApplicantIncome=col_b.number_input('Income of Applicant?')
CoapplicantIncome=col_a.number_input('Income of Co-Applicant? Leave if there isnt any Co Applicant.')
LoanAmount=col_b.number_input('Required Loan Amount (in thousands)?')
Loan_Amount_Term=col_a.number_input('Loan Term in months')
Credit_History=col_a.radio("Credit history Meets guidelines",('Yes', 'No'))
Property_Area=col_b.radio("What is the Type of Property",('Urban', 'Rural','Semi Urban'))#urban 2 #rural 0 #semi urban 1
LoanAmount_log=np.log(LoanAmount)





if st.button('Predict the loan approval'):
    # creating extra features
    TotalIncome=ApplicantIncome+CoapplicantIncome
    LoanPerIncome=LoanAmount/TotalIncome
    LoanPerTerm=LoanAmount/Loan_Amount_Term
    RepaymentRatio=(LoanPerTerm*1000)/TotalIncome

    #Encoding the data according to the original
    Gender=1 if Gender=='Male'else 0   
    Married =1 if Married else 0
    Education =0 if Education else 1
    Self_Employed =1 if Self_Employed else 1
    Credit_History =1 if Credit_History=='Yes' else 0
    if Property_Area=='Urban':
        Property_Area=2 
    elif Property_Area=='Rural':
        Property_Area=0
    else:
        Property_Area=1 
    data=np.array([Gender, Married, Dependents, Education,Self_Employed,ApplicantIncome, CoapplicantIncome, LoanAmount,Loan_Amount_Term, Credit_History, Property_Area,LoanAmount_log,TotalIncome,LoanPerIncome, LoanPerTerm, RepaymentRatio])
    data=data.reshape(1,-1)
    #print(data)        
    #st.write(data.shape)
    
    # Feature Scaling
    sc=load('std_scaler.bin')
    data=sc.transform(data)
    
    #applying the model to get the prediction
    model = pickle.load(open('logistic_regression.pkl', 'rb'))
    output= model.predict(data)
    st.write(output)
    print(type(output))








# More on the process
tab_titles=["Dataset Exploration","Data Cleaning", "Feature Generation", "Model Training"]
tabs=st.tabs(tab_titles)

with tabs[0]:
    st.header("Dataset Exploration")
    st.markdown("""Before delving deep into this project, we would neet to acquire insights on what the data 
    tells about the situation and the people applying. This would be very valuable in various areas like data cleaning, feature enginnering
    and explaining the results later.     
    """)

    st.write("""First, let's see what the dataset is all about, for which we list the columns of the dataset below.""")    
    st.write(train.columns.tolist())
    

    st.subheader("Single Variable analysis")
    st.markdown("Let us now see how the data given to us is distributed. Firstly, lets look at the loan status.")

    st.bar_chart(train['Loan_Status'].value_counts())
    st.markdown("In the above Graph, N denotes no and Y denotes Yes. The loan of 422 (around 69%) people out of 614 was approved. Now let us see more about other variables.")
    
    st.markdown("---")
    st.write("Now, lets Look at the other categorial variables in the dataset.")

    col1,col2=st.columns(2)


    col1.bar_chart(train['Gender'].value_counts(normalize=True))
    col1.write("About **80%** applicants in the dataset are male.")
    col1.bar_chart(train['Married'].value_counts(normalize=True))
    col1.write("Around **65%** of the applicants in the dataset are married.")
    
    col2.bar_chart(train['Self_Employed'].value_counts(normalize=True))
    col2.write("Around **15%** applicants in the dataset are self employed.")
    
    col2.bar_chart(train['Credit_History'].value_counts(normalize=True))
    col2.markdown("""A **1** in the dataset represents that the person has paid their debts before, 
    while 0 denotes that there were certain irregularities.Around 85% applicants have repaid their debts.""")
    st.markdown("---")
    
    col3,col4=st.columns(2)
    col3.bar_chart(train['Dependents'].value_counts(normalize=True))
    col3.write("Most of the applicants don’t have any dependents.")
    col4.bar_chart(train['Education'].value_counts(normalize=True))
    col4.write("Around 80% of the applicants are Graduate.")
    st.markdown("---")

with tabs[1]:
    st.write("There are a lot of missing values in the dataset. First, lets understand the number of missing values.")
    st.write(train.isnull().sum())
    st.markdown("""
    The missing data are of 2 types
    1. Numerical 
    2. Categorical

    ### Dealing with the Categorical Missing values

    The more obivous choice for imputation is through filling the numerical variables with the median, 
    while we could use the mode for the categorical variables, as the data missing is small for these columns.

    Gender, Married, Dependents, Credit History and Self_Employed Features are replaced *using mode* of the respective feature.

    Here, we could use some of the knowledge we get from EDA part to impute certain values.
    From the Bi variate Analysis before, we observe that the `Credit_History` parameter is a very good indicator that predicts the output variable.
    Thus, to improve our training, we can replace the missing `Credit_History` parameter based on the output.

    ```python
    train['Credit_History']=np.where((train['Credit_History'].isnull()) & (train['Loan_Status']=='Y'),1,train['Credit_History'])
    train['Credit_History']=np.where((train['Credit_History'].isnull()) & (train['Loan_Status']=='N'),0,train['Credit_History'])
    ```
    
    Even the Loan_Amount_Term is a categorical variable
        
    """)
    st.write(train['Loan_Amount_Term'].value_counts())
    st.markdown("""
     This above table shows us that the 360 months option is of the highest choosen one by the applicants. This, it makes sense to replace the 
     missing values in this column with the mode too.  
    
    """)
    
    st.markdown("""
    
    ### Dealing with Numerical Missing Values

    The only numerical feature that is missing is the `LoanAmount` variable, and we can use the median to fill out the missing values, as 
    the dats is skewed, and the mean would misrepresent the details.
    
    ```python
    train['LoanAmount_log'] = np.log(train['LoanAmount']) 
    ```
    
    ## Dealing with the text data

    We have information of some of the columns in text form, like 'yes' and 'no' and places like 'Dependents', where there is a mention of '3+'
    To eliminate all these, we need to Do 2 things
    1. Replace the '3+' to 3 in the 'Dependents' column, and change the type to int. This is because Dependents value matter, and is just not another class.
    2. Replace the Loan Status variable from 'N' and 'Y' to 0 and 1 to get proper results

    For the rest, we could use the label encoder from sklearn from processing to fill in the results.

   These steps clean almost all of the dataset, and makes it ready for use in sklearn models. Before that, we need to Do some feature extraction to 
    
    
    """)

with tabs[2]:
    st.markdown("""
    # Feature Generation

    As we are dealing with a loan approval in the scenario, we could use some of the basic knowledge we have on how loan payment works
    and how they approve loans. 

    Basically, if a person is given a loan, they would have to pay the money in installments over the years. This means that the
    income of the Applicant and /or coApplicant should be large enoung to cover the remainder of the expenses after covering this 
    loan payment. So, We can now create new features like 
    1. Ratio of Loan Amount and Income
    2. Ratio of Loan Amount over the Loan term
    3. Repayment Ratio

    This can be done in code simply as follows:

    ```python
    df_temp=training_dataset['ApplicantIncome']+training_dataset['CoapplicantIncome']
    training_dataset['Loan/Income']=training_dataset['LoanAmount']/df_temp
    training_dataset['Loan/Term']=training_dataset['LoanAmount']/training_dataset['Loan_Amount_Term']
    training_dataset['RepaymentRatio']=(training_dataset['Loan/Term']*1000)/df_temp

    ```
    
    
    
    
    
    
    
    """)

with tabs[3]:
    st.markdown("""
    
    ## Training the Model

    This is a direct classification task , where the output is eith 0 or 1. This is thus a binary classification task.
     

    
    
    
    
    
    
    
    """)









#model demo tab
# with tabs[4]:
#        st.write("Model Demo Tab")
    # #get inputs from user
    # Gender=st.selectbox('Gender of the applicant', ('Female', 'Male'))
    # Married=st.checkbox('Are you Married?')
    # Dependents=st.number_input('Number of Dependents?',step=1,format='%d')
    # Education=st.checkbox('Are you a College Graduate?')
    # Self_Employed=st.checkbox('Are you Self Employed?')
    # ApplicantIncome=st.number_input('Income of Applicant?')
    # CoapplicantIncome=st.number_input('Income of Co-Applicant? Leave if there isnt any Co Applicant.')
    # LoanAmount=st.number_input('Required Loan Amount (in thousands)?')
    # Loan_Amount_Term=st.number_input('Loan Term in months')
    # Credit_History=st.radio("Credit history Meets guidelines",('Yes', 'No'))
    # Property_Area=st.radio("What is the Type of Property",('Urban', 'Rural','Semi Urban'))#urban 2 #rural 0 #semi urban 1
    # LoanAmount_log=np.log(LoanAmount)

    # # creating extra features
    # TotalIncome=ApplicantIncome+CoapplicantIncome
    # LoanPerIncome=LoanAmount/TotalIncome
    # LoanPerTerm=LoanAmount/Loan_Amount_Term
    # RepaymentRatio=(LoanPerTerm*1000)/TotalIncome

    # #Encoding the data according to the original
    # Gender=1 if Gender=='Male'else 0   
    # Married =1 if Married else 0
    # Education =0 if Education else 1
    # Self_Employed =1 if Self_Employed else 1
    # Credit_History =1 if Credit_History=='Yes' else 0
    # if Property_Area=='Urban':
    #     Property_Area=2 
    # elif Property_Area=='Rural':
    #     Property_Area=0
    # else:
    #     Property_Area=1 


    # if st.button('Predict the loan approval'):
    #     data=np.array([Gender, Married, Dependents, Education,Self_Employed,ApplicantIncome, CoapplicantIncome, LoanAmount,Loan_Amount_Term, Credit_History, Property_Area,LoanAmount_log,TotalIncome,LoanPerIncome, LoanPerTerm, RepaymentRatio])
    #     data=data.reshape(1,-1)
    #     #print(data)        
    #     #st.write(data.shape)
        
    #     # Feature Scaling
    #     sc=load('std_scaler.bin')
    #     data=sc.transform(data)
        
    #     #applying the model to get the prediction
    #     model = pickle.load(open('logistic_regression.pkl', 'rb'))
    #     output= model.predict(data)
    #     st.write(output)
    #     print(type(output))















