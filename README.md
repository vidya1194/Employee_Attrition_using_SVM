# Employee Attrition Using SVM

## Project Overview

This project aims to predict employee attrition using various machine learning models, including Logistic Regression and Support Vector Machines (SVM) with different kernels. The project also includes data preprocessing, visualization, and model evaluation, saving the results in an organized manner.


## Project Structure

  - **main.py**: The main script to run the application.
  - **data/**: Contains the dataset used for training and testing.
    - **HR_Employee_Attrition.xlsx**: The raw dataset.
  - **logs/**: Contains logs generated during the execution.
    - **app.log**: Log file for tracking application events.
  - **src/**: Contains the core functionality.
    - **data/**: Data loading utilities.
    - **models/**: Scripts for training, evaluating, and predicting using the model.
      - **preprocess_partition/**: Scripts for data preprocessing and partitioning.
      - **visualization/**: Scripts for data visualization and exploratory analysis.
    - **utils/**: Utility functions such as logging setup and saving images.
  - **storage/**: Directory where generated images and results are saved.
  - **requirements.txt**: Lists the Python packages required to run the project.


## Dependencies
Plese see the requirement.txt file

## How to Run
To execute the project, navigate to your project directory and run the main.py script. Ensure that Python is installed and accessible via your command line:

python main.py
