import numpy as np
import pandas as pd

#Load the datasets
testing_dataset=pd.read_csv("test_lAUu6dG.csv") 
training_dataset=pd.read_csv("train_ctrUa4K.csv")
training_dataset_orig=training_dataset.copy()
testing_dataset_orig=testing_dataset.copy()

#Fixing Missing Values in Credit History.
training_dataset['Credit_History']=np.where((training_dataset['Credit_History'].isnull()) & (training_dataset['Loan_Status']=='Y'),1,training_dataset['Credit_History'])
training_dataset['Credit_History']=np.where((training_dataset['Credit_History'].isnull()) & (training_dataset['Loan_Status']=='N'),0,training_dataset['Credit_History'])