import bnlearn
from dataset import load_data, convert_to_ranges
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class IBMBayesianNetwork:
    def __init__(self):
        self.df = convert_to_ranges(load_data())
        self.edges = None
        self.model = self.create_edges()

        # Score-based structure learning with hillclimb search
        print('[Structure learning with hillclimb search, BIC score...]')
        self.bic_model = bnlearn.structure_learning.fit(self.df, methodtype='hc', scoretype='bic', verbose=1)
        print('[Structure learning with hillclimb search, K2 score...]')
        self.k2_model = bnlearn.structure_learning.fit(self.df, methodtype='hc', scoretype='k2', verbose=1)
        print('[Structure learning with hillclimb search, BDeu score...]')
        self.bdeu_model = bnlearn.structure_learning.fit(self.df, methodtype='hc', scoretype='bdeu', verbose=1)
        bnlearn.plot(self.model)
        # Comparison
        self.scores, self.adjmat_diff = bnlearn.compare_networks(self.bic_model, self.model)
        print('[Comparing your structure with BIC model...]')
        print('Accuracy ' + str(self.accuracy()))
        print('Precision ' + str(self.precision()))
        print('Recall ' + str(self.recall()))
        print('F-Score ' + str(self.f_score()))
        print('Error ' + str(self.error()))
        self.scores, self.adjmat_diff = bnlearn.compare_networks(self.k2_model, self.model)
        print('[Comparing your structure with K2 model...]')
        print('Accuracy ' + str(self.accuracy()))
        print('Precision ' + str(self.precision()))
        print('Recall ' + str(self.recall()))
        print('F-Score ' + str(self.f_score()))
        print('Error ' + str(self.error()))
        self.scores, self.adjmat_diff = bnlearn.compare_networks(self.bdeu_model, self.model)
        print('[Comparing your structure with BDeu model...]')
        print('Accuracy ' + str(self.accuracy()))
        print('Precision ' + str(self.precision()))
        print('Recall ' + str(self.recall()))
        print('F-Score ' + str(self.f_score()))
        print('Error ' + str(self.error()))

        self.bn = bnlearn.parameter_learning.fit(self.model, self.df, methodtype='maximumlikelihood', verbose=1)
        # bnlearn.plot(self.model)
        # bnlearn.print_CPD(self.bn)

    def create_edges(self):
        self.edges = [('EducationField', 'Department'),
                 ('Department', 'EnvironmentSatisfaction'),
                 ('Department', 'JobSatisfaction'),
                 ('Age', 'NumCompaniesWorked'),
                 ('Age', 'MonthlyIncome'),
                 ('Age', 'YearsAtCompany'),
                 ('Age', 'Education'),
                 ('YearsAtCompany', 'MonthlyIncome'),
                 ('MonthlyIncome', 'JobSatisfaction'),
                 ('EnvironmentSatisfaction', 'Attrition'),
                 ('JobSatisfaction', 'Attrition'),
                 ('WorkLifeBalance', 'Attrition'),
                 ('MaritalStatus', 'Attrition'),
                 ('DistanceFromHome', 'Attrition'),
                 ('Education', 'Attrition')]
        return bnlearn.make_DAG(self.edges, verbose=1)

    def error(self):
        return (self.scores[0,1]+self.scores[1,0])/(self.scores[0,0]+self.scores[0,1]+self.scores[1,0]+self.scores[1,1])

    def accuracy(self):
        return 1-self.error()

    def precision(self):
        return self.scores[0,0] / (self.scores[0,0] + self.scores[0,1])

    def recall(self):
        return self.scores[0,0] / (self.scores[0,0] + self.scores[1,1])

    def f_score(self):
        return (2 * self.recall() * self.precision()) / (self.recall() + self.precision())

    def query(self, variable, evidence):
        query = bnlearn.inference.fit(self.bn, variables=variable, evidence=evidence, verbose=1)
        return query.df.to_dict(orient="records")


def main():
    bn = IBMBayesianNetwork()

    variable = [
        'Department',
        'EnvironmentSatisfaction',
        'JobSatisfaction',
        'Age',
        'MonthlyIncome',
        'WorkLifeBalance',
        'MaritalStatus',
        'DistanceFromHome',
        'Education',
        'EducationField',
        'YearsAtCompany',
        'Attrition'
    ]
    variable_range = range(0, len(variable))

    attrition = [
        'Yes',
        'No'
    ]
    attrition_range = range(0, len(attrition))

    department = [
        'Human Resources',
        'Sales',
        'Research & Development'
    ]
    department_range = range(0, len(department))

    education_field = [
        'Life Sciences',
        'Medical',
        'Human Resources',
        'Technical Degree',
        'Other',
        'Marketing'
    ]
    education_field_range = range(0, len(education_field))

    marital_status = [
        'Single',
        'Married',
        'Divorced'
    ]
    marital_status_range = range(0, len(marital_status))

    age = [
        1,
        2,
        3,
        4
    ]
    age_range = range(0, len(age))

    environment_satisfaction = [
        1,
        2,
        3,
        4
    ]
    environment_sat_range = range(0, len(environment_satisfaction)+1)

    job_satisfaction = [
        1,
        2,
        3,
        4
    ]
    job_sat_range = range(0, len(job_satisfaction)+1)

    work_life_balance = [
        1,
        2,
        3,
        4
    ]
    balance_range = range(0, len(work_life_balance)+1)

    years_at_company = [
        1,
        2,
        3,
        4,
        5
    ]
    years_range = range(0, len(years_at_company)+1)

    education = [
        1,
        2,
        3,
        4,
        5
    ]
    edu_range = range(0, len(education)+1)

    distance_home = [
        1,
        2,
        3,
        4
    ]
    distance_range = range(0, len(distance_home)+1)

    monthly_income = [
        1,
        2,
        3,
        4
    ]
    income_range = range(0, len(monthly_income)+1)
    print('[Prediction]')
    for index, value in enumerate(variable):
        print(f'{index}: {value}')

    already_chosen = []
    variable_answ = None
    while variable_answ not in variable_range:
        variable_answ = int(input('Choose a variable to predict:'))
    already_chosen.append(variable_answ)

    stop = False
    evidences = { }
    while not stop:
        print('------------------------------------------------------------------')
        for index, value in enumerate(variable):
            print(f'{index}: {value}')
        evidence_answ = None
        print('You already chose ' + str(already_chosen))
        while evidence_answ not in variable_range:
                evidence_answ = int(input('Choose an evidence:'))
        already_chosen.append(evidence_answ)

        # Department
        if evidence_answ == 0:
            for index, value in enumerate(department):
                print(f'{index}: {value}')
            answ = None
            while answ not in department_range:
                answ = int(input('Choose a department:'))
            evidences['Department'] = department[answ]

        # EnvironmentSatisfaction
        elif evidence_answ == 1:
            answ = None
            print('1: Low \n2: Medium \n3: High \n4: Very high')
            while answ not in environment_sat_range:
                answ = int(input('Choose environment satisfaction value:'))
            evidences['EnvironmentSatisfaction'] = environment_satisfaction[answ-1]

        # JobSatisfaction
        elif evidence_answ == 2:
            answ = None
            print('1: Low \n2: Medium \n3: High \n4: Very high')
            while answ not in job_sat_range:
                answ = int(input('Choose job satisfaction value:'))
            evidences['JobSatisfaction'] = job_satisfaction[answ-1]

        # Age
        elif evidence_answ == 3:
            answ = None
            print('1: [18, 28] \n2: [29, 39] \n3: [40, 50] \n4: [51, 101]')
            while answ not in age_range:
                answ = int(input('Choose age range:'))
            evidences['Age'] = age[answ-1]

        # MonthlyIncome
        elif evidence_answ == 4:
            answ = None
            print('1: [1.000, 6.999] \n2: [7.000, 13.999] \n3: [14.000, 18.999] \n4: [19.000+]')
            while answ not in income_range:
                answ = int(input('Choose a monthly income range:'))
            evidences['MonthlyIncome'] = monthly_income[answ-1]

        # WorkLifeBalance
        elif evidence_answ == 5:
            answ = None
            print('1: Bad \n2: Good \n3: Better \n4: Best')
            while answ not in balance_range:
                answ = int(input('Choose a work-life balance value:'))
            evidences['WorkLifeBalance'] = work_life_balance[answ-1]

        # MaritalStatus
        elif evidence_answ == 6:
            for index, value in enumerate(marital_status):
                print(f'{index}: {value}')
            answ = None
            while answ not in marital_status_range:
                answ = int(input('Choose a marital status:'))
            evidences['MaritalStatus'] = marital_status[answ]

        # DistanceFromHome
        elif evidence_answ == 7:
            answ = None
            print('1: [1, 9] \n2: [10, 19] \n3: [20, 28] \n4: [29+]')
            while answ not in distance_range:
                answ = int(input('Choose a distance from home range:'))
            evidences['DistanceFromHome'] = distance_home[answ-1]

        # Education
        elif evidence_answ == 8:
            answ = None
            print('1: Below College \n2: College \n3: Bachelor \n4: Master \n5: Doctor')
            while answ not in edu_range:
                answ = int(input('Choose an education level:'))
            evidences['Education'] = education[answ-1]

        # EducationField
        elif evidence_answ == 9:
            for index, value in enumerate(education_field):
                print(f'{index}: {value}')
            answ = None
            while answ not in education_field_range:
                answ = int(input('Choose an education field:'))
            evidences['EducationField'] = education_field[answ]

        # YearsAtCompany
        elif evidence_answ == 10:
            answ = None
            print('1: [0, 9] \n2: [10, 19] \n3: [20, 29] \n4: [30, 39] \n5: [40, 101]')
            while answ not in years_range:
                answ = int(input('Choose a years range:'))
            evidences['YearsAtCompany'] = years_at_company[answ-1]

        # Attrition
        elif evidence_answ == 11:
            for index, value in enumerate(attrition):
                print(f'{index}: {value}')
            answ = None
            while answ not in attrition_range:
                answ = int(input('Choose attrition:'))
            evidences['Attrition'] = attrition[answ]

        flag = input('Done choosing evidence(s)? y-n')
        if flag == 'y':
            stop = True
        else:
            stop = False

    print('Chosen evidence(s): \n' + str(evidences))
    query = bn.query(variable=[variable[variable_answ]], evidence=evidences)
    print(query)


if __name__ == "__main__":
    main()