import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from src.utils.utils import setup_logging, save_image


logger = setup_logging()

def metrics_score(actual, predicted, kernel, type):
    try:
       
        cm = confusion_matrix(actual, predicted)
        plt.figure(figsize=(8,5))
        sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        
        print(classification_report(actual, predicted))
                
        # Save the confusion matrix as an image
        figure = plt.gcf()  # Get the current figure
        save_image(figure, f'{kernel}_{type}_confusion_matrix.png')

        # Show the plot
        plt.show()
        
        
    except Exception as e:
        logger.error(f"Error in metrics_score: {e}")
        raise e

def evaluate_model(model, x_train, x_test, y_train, y_test, kernel):
    try:
        # Training set evaluation
        y_pred_train = model.predict(x_train)
        print("Training Performance:")
        metrics_score(y_train, y_pred_train, kernel, 'Training')
        
        # Test set evaluation
        y_pred_test = model.predict(x_test)
        print("Test Performance:")
        metrics_score(y_test, y_pred_test, kernel ,'Test')
        
        logger.info('Model evaluation completed.')
    except Exception as e:
        logger.error(f"Error in evaluate_model: {e}")
        raise e
