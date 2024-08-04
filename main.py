import pandas as pd
from src.data.load_process import load_and_preprocess_data, split_data
from src.model.train_model import train_logistic_regression, train_svm
from src.data.visualization import visualize_numerical_data, visualize_categorical_data, visualize_grouped_means
from src.model.evaluate_model import evaluate_model
from src.utils.utils import setup_logging, save_image


def main():
    
    logger = setup_logging()
    
    try:
        # Load and preprocess the data       
        df, df_before, num_cols, cat_cols = load_and_preprocess_data('data/HR_Employee_Attrition.xlsx')
        
        visualize_numerical_data(df_before, num_cols)
        visualize_categorical_data(df_before, cat_cols)
        visualize_grouped_means(df_before, num_cols)
        
        # Split the data
        x_train, x_test, y_train, y_test = split_data(df)
        
        # Train Logistic Regression model
        lg_model = train_logistic_regression(x_train, y_train)
        evaluate_model(lg_model, x_train, x_test, y_train, y_test,'logistic')
        
        # Train SVM model with different kernels
        for kernel in ['linear', 'rbf', 'poly']:
            svm_model = train_svm(x_train, y_train, kernel=kernel)
            print(f"Evaluating SVM with {kernel} kernel:")
            evaluate_model(svm_model, x_train, x_test, y_train, y_test, kernel)
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
