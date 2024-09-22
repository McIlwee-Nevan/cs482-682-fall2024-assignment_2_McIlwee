import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,LinearRegression
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

class MyLogisticRegression:
    def __init__(self, dataset_num, perform_test):
        self.training_set = None
        self.test_set = None
        self.model_logistic = None
        self.model_linear = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.perform_test = perform_test
        self.dataset_num = dataset_num
        self.read_csv(self.dataset_num)

    def read_csv(self, dataset_num):
        if dataset_num == '1':
            train_dataset_file = 'train_q1_1.csv'
            test_dataset_file = 'test_q1_1.csv'
        elif dataset_num == '2':
            train_dataset_file = 'train_q1_2.csv'
            test_dataset_file = 'test_q1_2.csv'
        else:
            print("unsupported dataset number")
            
        self.training_set = pd.read_csv(train_dataset_file, sep=',', header=0).to_numpy()
        self.X_train = self.training_set[:,:2]
        self.y_train = self.training_set[:, 2]
        if self.perform_test:
            self.test_set = pd.read_csv(test_dataset_file, sep=',', header=0).to_numpy()
            self.X_test = self.test_set[:,:2]
            self.y_test = self.test_set[:, 2]
        
        
    def model_fit_linear(self):
        '''
        initialize self.model_linear here and call the fit function
        '''
        self.model_linear = LinearRegression(fit_intercept=True)
        self.model_linear.fit(self.X_train, self.y_train)
    
    def model_fit_logistic(self):
        '''
        initialize self.model_logistic here and call the fit function
        '''

        self.model_logistic = LogisticRegression()
        self.model_logistic.fit(self.X_train, self.y_train)
    
    def model_predict_linear(self):
        '''
        Calculate and return the accuracy, precision, recall, f1, support of the model.
        '''
        self.model_fit_linear()
        accuracy = 0.0
        precision, recall, f1, support = np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0])
        assert self.model_linear is not None, "Initialize the model, i.e. instantiate the variable self.model_linear in model_fit_linear method"
        assert self.training_set is not None, "self.read_csv function isn't called or the self.trianing_set hasn't been initialized "
        
        if self.X_test is not None:
            # perform prediction here
            y_pred = self.model_linear.predict(self.X_test)
            y_pred = np.sign(np.sign(y_pred - 0.5) + 1)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred)
            
            print("\nClassification Report:\n", classification_report(self.y_test, y_pred))
        
        assert precision.shape == recall.shape == f1.shape == support.shape == (2,), "precision, recall, f1, support should be an array of shape (2,)"
        return [accuracy, precision, recall, f1, support]

    def model_predict_logistic(self):
        '''
        Calculate and return the accuracy, precision, recall, f1, support of the model.
        '''
        self.model_fit_logistic()
        accuracy = 0.0
        precision, recall, f1, support = np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0])
        assert self.model_logistic is not None, "Initialize the model, i.e. instantiate the variable self.model_logistic in model_fit_logistic method"
        assert self.training_set is not None, "self.read_csv function isn't called or the self.trianing_set hasn't been initialized "
        
        if self.X_test is not None:
            # perform prediction here
            y_pred = self.model_logistic.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred)
            
            print("\nClassification Report:\n", classification_report(self.y_test, y_pred))
        
        assert precision.shape == recall.shape == f1.shape == support.shape == (2,), "precision, recall, f1, support should be an array of shape (2,)"
        return [accuracy, precision, recall, f1, support]


    def make_plot(self):
        if self.dataset_num == '1':
            x_min, x_max = self.X_test[:, 0].min() - 5, self.X_test[:, 0].max() + 5
            y_min, y_max = self.X_test[:, 1].min() - 5, self.X_test[:, 1].max() + 5
        elif self.dataset_num == '2':
            x_min, x_max = self.X_test[:, 0].min() - 0.5, self.X_test[:, 0].max() + 0.5
            y_min, y_max = self.X_test[:, 1].min() - 0.5, self.X_test[:, 1].max() + 0.5

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))
        
        Z_linear = self.model_linear.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_linear = np.sign(np.sign(Z_linear - 0.5) + 1)
        Z_linear = Z_linear.reshape(xx.shape)

        Z_logistic = self.model_logistic.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_logistic = Z_logistic.reshape(xx.shape)

        figure, ([linear_plot, logistic_plot]) = plt.subplots(1,2)
        figure.suptitle('Data Set %d' % int(self.dataset_num))
        
        linear_plot.contourf(xx, yy, Z_linear, alpha=0.6)
        linear_plot.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, edgecolors='k', marker='o', s=50)
        linear_plot.set_title('Linear Regression')
        linear_plot.set_xlabel('Test Score 1')
        linear_plot.set_ylabel('Test Score 2')

        logistic_plot.contourf(xx, yy, Z_logistic, alpha=0.6)
        logistic_plot.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, edgecolors='k', marker='o', s=50)
        logistic_plot.set_title('Logistic Regression')
        logistic_plot.set_xlabel('Test Score 1')
        logistic_plot.set_ylabel('Test Score 2')

        plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Regression')
    parser.add_argument('-d','--dataset_num', type=str, default = "1", choices=["1","2"], help='string indicating datset number. For example, 1 or 2')
    parser.add_argument('-t','--perform_test', action='store_true', help='boolean to indicate inference')
    args = parser.parse_args()
    classifier = MyLogisticRegression(args.dataset_num, args.perform_test)
    acc_linear = classifier.model_predict_linear()
    acc_logistic = classifier.model_predict_logistic()
    classifier.make_plot()
    