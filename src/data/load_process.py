import pandas as pd
from src.utils.utils import setup_logging

logger = setup_logging()

def load_and_preprocess_data(file_path):
    try:
        # Read the dataset
        df = pd.read_excel(file_path)
        logger.info('Data loaded successfully.')
        
        print(df.sample(5))
        
        # Dropping unnecessary columns
        df = df.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
        logger.info('Unnecessary columns dropped.')
        
        print(df.info())
        
        # Creating numerical and categorical columns
        num_cols = [
            'DailyRate', 'Age', 'DistanceFromHome', 'MonthlyIncome', 'MonthlyRate',
            'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany', 'NumCompaniesWorked',
            'HourlyRate', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
            'TrainingTimesLastYear'
        ]
        cat_cols = [
            'Attrition', 'OverTime', 'BusinessTravel', 'Department', 'Education', 'EducationField',
            'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'StockOptionLevel', 
            'Gender', 'PerformanceRating', 'JobInvolvement', 'JobLevel', 'JobRole', 'MaritalStatus', 
            'RelationshipSatisfaction'
        ]
        
        df_before = df
        
        # Creating list of dummy columns
        to_get_dummies_for = ['BusinessTravel', 'Department','Education', 
                              'EducationField','EnvironmentSatisfaction', 
                              'Gender',  'JobInvolvement','JobLevel', 
                              'JobRole', 'MaritalStatus']

        # Creating dummy variables
        df = pd.get_dummies(data=df, columns=to_get_dummies_for, drop_first=True)
        logger.info('Dummy variables created.')
        
        # Mapping overtime and attrition
        dict_OverTime = {'Yes': 1, 'No': 0}
        dict_attrition = {'Yes': 1, 'No': 0}
        
        df['OverTime'] = df.OverTime.map(dict_OverTime)
        df['Attrition'] = df.Attrition.map(dict_attrition)
        logger.info('OverTime and Attrition mapped.')
        
        return df, df_before, num_cols, cat_cols
    except Exception as e:
        logger.error(f"Error in load_and_preprocess_data: {e}")
        
        raise e

def split_data(df):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    try:
        Y = df.Attrition
        X = df.drop(columns=['Attrition'])
        
        # Scaling the data
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        logger.info('Data scaled successfully.')
        
        # Splitting the data
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)
        logger.info('Data split into train and test sets.')
        
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error in split_data: {e}")
        raise e
