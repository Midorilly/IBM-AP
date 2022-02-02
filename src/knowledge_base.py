import pytholog as pl
import dataset as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

hr_kb = pl.KnowledgeBase('IBM-HR')
df = dt.load_data()
kb = []

# 0-1
data = dt.get_educationfield_department(df)
for d in data:
    kb.append(f"suitable_department({d['EducationField'].lower()},{d['Department'].lower()})")
    kb.append(f"working_field({d['Department'].lower()},{d['EducationField'].lower()})")

# 2
data = dt.get_avg_monthlyincome(df)
for key, value in data.items():
    kb.append(f"age_average_income({key},{value})")

# 3
data = dt.get_department_maritalstatus(df)
for d in data:
    kb.append(f"department_marital_status({d['Department'].lower()},{d['MaritalStatus'].lower()},{d['percentage']})")

# 4
data = dt.get_avg_environmentsatisfaction(df)
for key, value in data.items():
    kb.append(f"average_environment_satisfaction({key.lower()},{value})")

# 5
data = dt.get_avg_jobsatisfaction(df)
for key, value in data.items():
    kb.append(f"average_job_satisfaction({key.lower()},{value})")

# 6
data = dt.get_avg_attrition_jobsatisfaction(df)
for d in data:
    kb.append(f"average_job_attrition({d['Department'].lower()},{d['Attrition'].lower()},{d['JobSatisfaction']})")

# 7
data = dt.get_department_attrition(df)
for d in data:
    kb.append(f"department_attrition_percentage({d['Department'].lower()},{d['Attrition'].lower()},{d['percentage']})")

hr_kb(kb)


def main():
    query = [
        'suitable_department',
        'working_field',
        'age_average_income',
        'department_marital_status',
        'average_environment_satisfaction',
        'average_job_satisfaction',
        'average_job_attrition',
        'department_attrition_percentage',
    ]
    query_range = range(0, len(query))

    department = [
        'human resources',
        'sales',
        'research & development',
        'Q'
    ]
    department_range = range(0, len(department))

    education_field = [
        'life sciences',
        'medical',
        'human resources',
        'technical degree',
        'other',
        'marketing',
        'Q'
    ]
    education_field_range = range(0, len(education_field))

    marital_status = [
        'single',
        'married',
        'divorced'
    ]
    marital_status_range = range(0, len(marital_status))

    attrition = [
        'yes',
        'no'
    ]
    attrition_range = range(0, len(attrition))

    age_range = range(18, 101)

    for index, value in enumerate(query):
        print(f'{index}: {value}')

    query_answ = None
    while query_answ not in query_range:
        query_answ = int(input('Choose a query:'))

    if query_answ == 0:
        for index, value in enumerate(education_field):
            print(f'{index}: {value}')
        ed_field_answ = None
        while ed_field_answ not in education_field_range:
            ed_field_answ = int(input('Choose an education field'))

        for index, value in enumerate(department):
            print(f'{index}: {value}')
        department_answ = None
        while department_answ not in department_range:
            department_answ = int(input('Choose a department:'))

        print(hr_kb.query(pl.Expr(f"{query[query_answ]}({education_field[ed_field_answ]},{department[department_answ]})")))

    elif query_answ == 1:
        for index, value in enumerate(department):
            print(f'{index}: {value}')
        department_answ = None
        while department_answ not in department_range:
            department_answ = int(input('Choose a department:'))

        for index, value in enumerate(education_field):
            print(f'{index}: {value}')
        ed_field_answ = None
        while ed_field_answ not in education_field_range:
            ed_field_answ = int(input('Choose an education field'))

        print(hr_kb.query(pl.Expr(f"{query[query_answ]}({department[department_answ]},{education_field[ed_field_answ]})")))

    elif query_answ == 2:
        age_answ = None
        while age_answ not in age_range:
            age_answ = int(input('Choose an age between 18 and 101:'))

        print(hr_kb.query(pl.Expr(f"{query[query_answ]}({age_answ},Q)")))

    elif query_answ == 3:
        for index, value in enumerate(department):
            print(f'{index}: {value}')
        department_answ = None
        while department_answ not in department_range:
            department_answ = int(input('Choose a department:'))

        for index, value in enumerate(marital_status):
            print(f'{index}: {value}')
        status_answ = None
        while status_answ not in marital_status_range:
            status_answ = int(input('Choose a marital status:'))

        print(hr_kb.query(pl.Expr(f"{query[query_answ]}({department[department_answ]},{marital_status[status_answ]},Q)")))

    elif query_answ == 4 or query_answ == 5:
        for index, value in enumerate(department):
            print(f'{index}: {value}')
        department_answ = None
        while department_answ not in department_range:
            department_answ = int(input('Choose a department:'))

        print(hr_kb.query(pl.Expr(f"{query[query_answ]}({department[department_answ]},Q)")))

    elif query_answ == 6 or query_answ == 7:
        for index, value in enumerate(department):
            print(f'{index}: {value}')
        department_answ = None
        while department_answ not in department_range:
            department_answ = int(input('Choose a department:'))

        for index, value in enumerate(attrition):
            print(f'{index}: {value}')
        attrition_answ = None
        while attrition_answ not in attrition_range:
            attrition_answ = int(input('Choose attrition:'))

        print(hr_kb.query(pl.Expr(f"{query[query_answ]}({department[department_answ]},{attrition[attrition_answ]},Q)")))


if __name__ == "__main__":
    main()
