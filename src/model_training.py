
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
class model_training:
    
    def __init__(self):
        self.model = None
    
    def train_model_RandomForest(self, x_train, y_train):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42 , max_depth=5
                                            )
        self.model.fit(x_train, y_train)
    def train_model_LogisticRegression(self, x_train, y_train):
        self.model = LogisticRegression()
        self.model.fit(x_train, y_train)  
    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def evaluate_model(self, x_test, y_test):
        
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        pred = self.predict(x_test)
        print(metrics.classification_report(y_test, pred))
        print(metrics.confusion_matrix(y_test, pred))
        
   