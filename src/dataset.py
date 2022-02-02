import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import bnlearn


# Import dataset
def load_data():
    df = pd.read_csv("dataset/IBM.csv")
    # Check for empty elements
    df = df.dropna()
    # Check for duplicate rows
    df.drop_duplicates(inplace=True)
    # print(odf)
    return df


def convert_to_ranges(df):
    df_c = df.copy()
    # Convert 'Age' in intervals between 1 and 4
    df_c.loc[(df_c['Age'] >= 18) & (df_c['Age'] <= 28), ['Age']] = 1
    df_c.loc[(df_c['Age'] >= 29) & (df_c['Age'] <= 39), ['Age']] = 2
    df_c.loc[(df_c['Age'] >= 40) & (df_c['Age'] <= 50), ['Age']] = 3
    df_c.loc[(df_c['Age'] >= 51) & (df_c['Age'] <= 101), ['Age']] = 4

    # Convert 'DistanceFromHome' in intervals between 1 and 4
    df_c.loc[(df_c['DistanceFromHome'] >= 1) & (df_c['DistanceFromHome'] <= 9), ['DistanceFromHome']] = 1
    df_c.loc[(df_c['DistanceFromHome'] >= 10) & (df_c['DistanceFromHome'] <= 19), ['DistanceFromHome']] = 2
    df_c.loc[(df_c['DistanceFromHome'] >= 20) & (df_c['DistanceFromHome'] <= 28), ['DistanceFromHome']] = 3
    df_c.loc[(df_c['DistanceFromHome'] >= 29), ['DistanceFromHome']] = 4

    # Convert 'MonthlyIncome' in intervals between 1 and 4
    df_c.loc[(df_c['MonthlyIncome'] >= 1000) & (df_c['MonthlyIncome'] <= 6999), ['MonthlyIncome']] = 1
    df_c.loc[(df_c['MonthlyIncome'] >= 7000) & (df_c['MonthlyIncome'] <= 13999), ['MonthlyIncome']] = 2
    df_c.loc[(df_c['MonthlyIncome'] >= 14000) & (df_c['MonthlyIncome'] <= 18999), ['MonthlyIncome']] = 3
    df_c.loc[(df_c['MonthlyIncome'] >= 19000), ['MonthlyIncome']] = 4

    # Convert 'NumCompaniesWorked' in intervals between 1 and 4
    df_c.loc[(df_c['NumCompaniesWorked'] >= 0) & (df_c['NumCompaniesWorked'] <= 2), ['NumCompaniesWorked']] = 1
    df_c.loc[(df_c['NumCompaniesWorked'] >= 3) & (df_c['NumCompaniesWorked'] <= 5), ['NumCompaniesWorked']] = 2
    df_c.loc[(df_c['NumCompaniesWorked'] >= 6) & (df_c['NumCompaniesWorked'] <= 8), ['NumCompaniesWorked']] = 3
    df_c.loc[(df_c['NumCompaniesWorked'] >= 9), ['NumCompaniesWorked']] = 4

    # Convert 'YearsAtCompany' in intervals between 1 and 5
    df_c.loc[(df_c['YearsAtCompany'] >= 0) & (df_c['YearsAtCompany'] <= 9), ['YearsAtCompany']] = 1
    df_c.loc[(df_c['YearsAtCompany'] >= 10) & (df_c['YearsAtCompany'] <= 19), ['YearsAtCompany']] = 2
    df_c.loc[(df_c['YearsAtCompany'] >= 20) & (df_c['YearsAtCompany'] <= 29), ['YearsAtCompany']] = 3
    df_c.loc[(df_c['YearsAtCompany'] >= 30) & (df_c['YearsAtCompany'] <= 39), ['YearsAtCompany']] = 4
    df_c.loc[(df_c['YearsAtCompany'] >= 40) & (df_c['YearsAtCompany'] <= 101), ['YearsAtCompany']] = 5

    return df_c

# One-Hot Binary Encoding
def convert_one_hot_encoding(df):
    df_c = df.copy()
    df_c = pd.concat([df_c, pd.get_dummies(df_c.Age, prefix='AGE')], axis=1)
    df_c = df_c.drop(['Age'], axis=1)

    df_c = pd.concat([df_c, pd.get_dummies(df_c.Attrition, prefix='ATTRITION')], axis=1)
    df_c = df_c.drop(['Attrition'], axis=1)

    df_c = pd.concat([df_c, pd.get_dummies(df_c.Department, prefix='D')], axis=1)
    df_c = df_c.drop(['Department'], axis=1)

    df_c = pd.concat([df_c, pd.get_dummies(df_c.DistanceFromHome, prefix='df_cH')], axis=1)
    df_c = df_c.drop(['DistanceFromHome'], axis=1)

    df_c = pd.concat([df_c, pd.get_dummies(df_c.Education, prefix='E')], axis=1)
    df_c = df_c.drop(['Education'], axis=1)

    df_c = pd.concat([df_c, pd.get_dummies(df_c.EducationField, prefix='EF')], axis=1)
    df_c = df_c.drop(['EducationField'], axis=1)

    df_c = pd.concat([df_c, pd.get_dummies(df_c.EnvironmentSatisfaction, prefix='ES')], axis=1)
    df_c = df_c.drop(['EnvironmentSatisfaction'], axis=1)

    df_c = pd.concat([df_c, pd.get_dummies(df_c.JobSatisfaction, prefix='JS')], axis=1)
    df_c = df_c.drop(['JobSatisfaction'], axis=1)

    df_c = pd.concat([df_c, pd.get_dummies(df_c.MaritalStatus, prefix='MS')], axis=1)
    df_c = df_c.drop(['MaritalStatus'], axis=1)

    df_c = pd.concat([df_c, pd.get_dummies(df_c.MonthlyIncome, prefix='MI')], axis=1)
    df_c = df_c.drop(['MonthlyIncome'], axis=1)

    df_c = pd.concat([df_c, pd.get_dummies(df_c.NumCompaniesWorked, prefix='NCW')], axis=1)
    df_c = df_c.drop(['NumCompaniesWorked'], axis=1)

    df_c = pd.concat([df_c, pd.get_dummies(df_c.WorkLifeBalance, prefix='WLB')], axis=1)
    df_c = df_c.drop(['WorkLifeBalance'], axis=1)

    df_c = pd.concat([df_c, pd.get_dummies(df_c.YearsAtCompany, prefix='YAC')], axis=1)
    df_c = df_c.drop(['YearsAtCompany'], axis=1)
    return df_c


# Plot data distributions using kernel density estimation
def plot_kde(df):
    plt.figure('Data Distribution')
    g = sns.pairplot(df, hue=['Attrition'], height=4)
    g.map_upper(sns.kdeplot, levels=1, color=".2")
    plt.show()


# Plot correlation matrix
def plot_corr_matrix(df):
    plt.figure(figsize=(50, 20))
    plt.title('Features Correlation matrix')
    sns.heatmap(df.corr(), vmin=-1, vmax=1, center=0, annot=True)
    plt.show()


def get_educationfield_department(df):
    return df[['EducationField', 'Department']].drop_duplicates().to_dict(orient="records")
    #return df.groupby(['Department', 'EducationField'], sort=False)['Department', 'EducationField'].apply(lambda x: list(np.unique(x)))


def get_avg_monthlyincome(df):
    return df.groupby(['Age'])['MonthlyIncome'].mean().to_dict()


def get_department_maritalstatus(df):
    status = df.groupby(['Department', 'MaritalStatus'])['MaritalStatus'].count().reset_index(name='counts')
    department = df.groupby(['Department'])['Department'].count().reset_index(name='counts')
    merged = pd.merge(status, department, on='Department', how='inner')
    merged['percentage'] = merged['counts_x']/merged['counts_y'] * 100
    merged.drop(['counts_x', 'counts_y'], axis=1, inplace=True)
    return merged.to_dict('records')


def get_avg_environmentsatisfaction(df):
    return df.groupby(['Department'])['EnvironmentSatisfaction'].mean().to_dict()


def get_avg_jobsatisfaction(df):
    return df.groupby(['Department'])['JobSatisfaction'].mean().to_dict()


def get_avg_attrition_jobsatisfaction(df):
    res = df.groupby(['Department', 'Attrition'])['JobSatisfaction'].mean()
    return res.to_frame().reset_index().to_dict('records')


def get_department_attrition(df):
    attrition = df.groupby(['Department', 'Attrition'])['Attrition'].count().reset_index(name='counts')
    department = df.groupby(['Department'])['Department'].count().reset_index(name='counts')
    merged = pd.merge(attrition, department, on='Department', how='inner')
    merged['percentage'] = merged['counts_x'] / merged['counts_y'] * 100
    merged.drop(['counts_x', 'counts_y'], axis=1, inplace=True)
    return merged.to_dict('records')

