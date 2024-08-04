import logging
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from src.utils.utils import setup_logging


logger = setup_logging()

def train_logistic_regression(x_train, y_train):
    try:
        lg = LogisticRegression()
        lg.fit(x_train, y_train)
        logger.info('Logistic Regression model trained successfully.')
        return lg
    except Exception as e:
        logger.error(f"Error in train_logistic_regression: {e}")
        raise e

def train_svm(x_train, y_train, kernel='linear', degree=3):
    try:
        svm = SVC(kernel=kernel, degree=degree)
        model = svm.fit(x_train, y_train)
        logger.info(f'SVM model with {kernel} kernel trained successfully.')
        return model
    except Exception as e:
        logger.error(f"Error in train_svm: {e}")
        raise e
