import pandas  as pd
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
    #Detta är en gemensam metod för att göra förutsägelser med den tränade modellen.
    # Den tar x_test som input och returnerar modellens förutsägelser genom att anropa predict-metoden på modellen. 
    # Detta gör det möjligt att enkelt generera förutsägelser på testdata eller annan data
    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def evaluate_model(self, x_test, y_test):
        pred = self.predict(x_test)
        print(metrics.classification_report(y_test, pred))
        print(metrics.confusion_matrix(y_test, pred))
    
    # Denna metod utvärderar den tränade modellen på träningsdata. Den gör 
    # förutsägelser på x_train och jämför dem med y_train för att generera en klassificeringsrapport och en förvirringsmatris, 
    # som sedan skrivs ut. Detta kan hjälpa till att bedöma modellens prestanda på träningsdata.
     
    def evalute_trained_model(self, x_train, y_train):
      pred = self.predict(x_train)
      print(metrics.classification_report(y_train, pred))
      print(metrics.confusion_matrix(y_train, pred))
    
    def evalute(self, x_test, y_test):
        pred=self.predict(x_test)
        print("accuracy:"+str(metrics.accuracy_score(y_test, pred)))
        print("precision:"+str(metrics.precision_score(y_test, pred)))
        print("f1-score:"+str(metrics.f1_score(y_test, pred)))
        print("recall:"+str(metrics.recall_score(y_test, pred)))
        
    def feature_importance_Forestrandom(self,X):
           
            importances = self.model.feature_importances_
            feature_names = X.columns

            print("len(feature_names):", len(feature_names))
            print("len(importances):", len(importances))
            print("feature names:", list(feature_names))

            df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values(by="importance", ascending=False)

            return df
