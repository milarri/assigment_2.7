"""
Created on Mon Jan 4 10:07:04 2025

@author: Martina Ilarri
"""
#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Loading the Excel file
file_path = r"C:\Users\Toshiba\Documents\data\UCMF.xls"
xls = pd.ExcelFile(file_path)

# Check the sheet names to understand the structure of the file
print("Sheet names:", xls.sheet_names)

# Loading the data from the first sheet (or any specific sheet by name)
df = pd.read_excel(file_path, sheet_name=xls.sheet_names[0])
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print('original:', df.shape)
print(df.describe(include='all'))

# providing a Histogram with the target variables (varibles that were considered relevant to explore), 
# In order to better evaluate the NAN values (-1), several variables (considered not relevant) were excluded.
X = df.drop(columns=['ID', 'Atendimento', 'SEXO','DN', 'Convenio','IMC', 'PA SISTOLICA', 
                     'PA DIASTOLICA', 'PPA', 'B2', 'SOPRO', 'FC', 'HDA 1', 'HDA2',
                     'MOTIVO1', 'MOTIVO2'])
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,15), constrained_layout=True)
plots = axes.flatten()
for idx, i in enumerate(X):
    plots[idx].set_title(i)
    try:
        plots[idx].hist(X[i].values)
    except:
        try:
            plots[idx].hist(pd.factorize(X[i])[0])
        except:
            continue
plt.show()

# looking for the outliers in the target variables 
def outlier(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3-q1
    low = q1-1.5*iqr
    high = q3+1.5*iqr
    return df.loc[(df > low) & (df < high)]

for i in X:
    try:
        out = outlier(X[i])        
        print(len(X[i])-len(out), 'outliers in', i)
    except:
        try:
            data = pd.DataFrame(pd.factorize(X[i])[0], columns=[i])
            out = outlier(data[i])
            print(len(X[i])-len(out), 'outliers in', i)
        except:
            continue
        

# Looked for the NAN values in the target variables. The NAN values were dropped. 
# In these cases, the columns and rows of the target variables, 
# that contained more than 50% missing values were removed (method below).
df = df.drop(columns=['ID', 'Atendimento', 'SEXO', 'DN', 'Convenio','IMC', 'PA SISTOLICA', 
                     'PA DIASTOLICA', 'PPA', 'B2', 'SOPRO', 'FC', 'HDA 1', 'HDA2',
                     'MOTIVO1', 'MOTIVO2'])
df = df.dropna(subset=['NORMAL X ANORMAL'])
for col in df.columns:
    nan = len(df[col]) - df[col].count()
    p = 100 * nan / len(df[col])
    print('{} NaN: {}%'.format(col, int(p)))
    if p > 50:
        df = df.drop(columns=[col])
        
for index, row in df.iterrows():
    nan = len(row) - row.count()
    p = 100 * nan / len(row)
    if p > 50:
        df = df.drop(index=index)
print('dropped:', df.shape)
print(df.iloc[0])

# Age columns contained "#VALUE!" values, and to fix it, values were replaced by NaN values, which can be seen below.
# Also, although 'int' type describe better the age values, to avoid any issue related to NaN, 'float' was used to describe this type of data. 
print(df.IDADE.unique())
df.loc[df.IDADE=='#VALUE!', 'IDADE'] = np.nan
df.IDADE = df.IDADE.astype(float)

# Also, some inconsistent values were found in 'Weight'and'Height'columns, 
# like negative values or impossible values. At this stage these values were dropped. 
df = df.drop(df[(df.Peso < 1) & (df.Altura < 30)].index)
df = df.drop(df[(df.IDADE < 0)].index)
print('dropped:', df.shape)

# Some values in the dataset can be considered redundant, i.e, different values in a certain variable convey the same meaning. 
#To make this correction easy, the function 'unique' was used to show all possible values in each variable.
#Then, a common pattern was followed to describe redundant values.
#Afterwards, the function 'factorize' was applied to discretize values between 0 and N.By default the NaN values were replaced by -1 using this function.
#After the method applied, is used as type float that works well with NAN values and the -1 instances was replaced to NaN.

print(df.PULSOS.unique())
df.loc[df.PULSOS=='Outro', 'PULSOS'] = np.nan
df.PULSOS = df.PULSOS.replace('NORMAIS','Normais').replace('AMPLOS','Amplos')
df.PULSOS = df.PULSOS.replace('Diminuídos ','Diminuídos').replace('Femorais diminuidos','Diminuídos')
df.PULSOS = pd.factorize(df.PULSOS)[0].astype(float)
df.loc[df.PULSOS==-1, 'PULSOS'] = np.nan
print(df.PULSOS.unique())

print(df['NORMAL X ANORMAL'].unique())
df['NORMAL X ANORMAL'] = df['NORMAL X ANORMAL'].replace('anormal','Anormal').replace('Normais','Normal')
print(df['NORMAL X ANORMAL'].unique())


#After some data preprocessing, outlier detection, and cleaning of the dataset it was considered anlyzing some of the target variables.

# Analyzing the 'NORMAL X ANORMAL' column that included healthy and non normal pacients.
if 'NORMAL X ANORMAL' in df.columns:
    normal_anormal_counts = df['NORMAL X ANORMAL'].value_counts()
    print("\nDistribution of NORMAL X ANORMAL:")
    print(normal_anormal_counts)

    # Bar Plot
    plt.figure(figsize=(8, 6))
    normal_anormal_counts.plot(kind='bar', color=['#8BC34A', '#FF5722'])
    plt.title('Distribution of normal vs abnormal patients')
    plt.xlabel('Status')
    plt.ylabel('Count')
# Customize x-axis labels
    plt.xticks(ticks=[0, 1], labels=['Normal', 'Abnormal'], rotation=0)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Pie Chart
    plt.figure(figsize=(6, 6))
    normal_anormal_counts.plot(kind='pie', 
                           autopct='%1.1f%%', 
                           labels=['Normal Patients', 'Abnormal Patients'], 
                           colors=['#8BC34A', '#FF5722'])
    plt.title('Proportion of normal vs abnormal patients')
    plt.ylabel('')  # Remove default ylabel
    plt.tight_layout()
    plt.show()
    

import matplotlib.pyplot as plt

# Making plots of the target variables (Age, Height, Weight) with the pacients information.

# Mapping of column names to descriptive names
variable_names = {
    'IDADE': 'Age',
    'Altura': 'Height (cm)',
    'Peso': 'Weight (kg)'
}

columns_to_plot = ['IDADE', 'Altura', 'Peso']

for col in columns_to_plot:
    if col not in df.columns:
        print(f"Column '{col}' not found in the dataset.")
        continue

# Histograms for numeric columns
numeric_columns = ['IDADE', 'Altura', 'Peso']
for col in numeric_columns:
    if col in df.columns:
        plt.figure(figsize=(8, 6))
        plt.hist(df[col].dropna(), bins=15, color='skyblue', edgecolor='black')
        plt.title(f'Histogram for {variable_names.get(col, col)}')
        plt.xlabel(variable_names.get(col, col))
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


