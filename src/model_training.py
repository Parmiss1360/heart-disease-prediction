import pandas  as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
class model_training:
    
    def __init__(self):
        self.model = None
    
    def train_model_RandomForest(self, x_train, y_train):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42 , max_depth=5
                                            )
        self.model.fit(x_train, y_train)
    
    
    #I denna metod tränas en XGBoost-modell på träningsdata (x_train och y_train)
    # med specifika hyperparametrar, inklusive n_estimators (antalet träd i modellen), random_state (för att säkerställa reproducerbarhet) och max_depth (maximalt djup på varje träd)
    # Efter träningen kan modellen användas för att göra förutsägelser på testdata eller annan data.
    # (för reproducerbarhet) och max_depth (maximalt djup på varje träd). Efter träningen kan modellen användas för att göra förutsägelser på testdata eller annan data.  
    def train_model_XGBoost(self, x_train, y_train):
        self.model = xgb.XGBClassifier(
                                      random_state=123
                                     , max_depth=5, subsample=1.0, learning_rate=0.2,gamma=0.6)
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
    
    # den här metoden beräknar och skriver ut olika utvärderingsmått för modellen, 
    # inklusive noggrannhet (accuracy), precision, F1-score och recall.
    def evalute(self, x_test, y_test):
        pred=self.predict(x_test)
        return {
        "accuracy": metrics.accuracy_score(y_test, pred),
        "precision": metrics.precision_score(y_test, pred),
        "f1": metrics.f1_score(y_test, pred),
        "recall": metrics.recall_score(y_test, pred)
    }
# den här metoden beräknar och skriver ut olika feature_importance_Forestrandom för modellen
# med hjälp av attributet feature_importances_ som finns i RandomForestClassifier. 
# Den skapar en DataFrame som innehåller varje funktions namn och dess motsvarande vikt, '
# sorterar den efter vikt i fallande ordning och returnerar den. 
# Detta kan hjälpa till att identifiera vilka funktioner som är mest betydelsefulla för modellens prediktioner.
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
