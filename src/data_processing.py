from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
class Data_processing:
    
#Här kan du lägga till mer initialisering om det behövs
    def __init__(self):
       
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.lower_bound = None
        self.upper_bound = None
        
        
    # Denna metod lägger till en ny kolumn 'oldpeak_sqrt' som är kvadratroten av 'oldpeak' 
    #vilket kan hjälpa till att normalisera fördelningen av 'oldpeak' och minska effekten av outliers.
    def add_extra_sqrt_column(self,data):
        data['oldpeak_sqrt'] = np.sqrt(data['oldpeak'])
        return data

   
        # Denna metod hanterar outliers i kolumnerna 'trestbps', 'chol' och 'thalach' genom att använda IQR-metoden för att bestämma nedre och övre gränser.
        # Om fit är True, beräknar den gränserna baserat på träningsdata och sparar dem som attribut. 
        # Om fit är False, använder den de sparade gränserna för att hantera outliers i testdata.
    
    def handle_outliers(self,data,  fit=False):
        cols = ['trestbps', 'chol','thalach']

        if fit:
            Q1 = data[cols].quantile(0.25)
            Q3 = data[cols].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bound = Q1 - 1.5 * IQR
            self.upper_bound = Q3 + 1.5 * IQR

        data[cols] = np.where(
            data[cols] < self.lower_bound,
            self.lower_bound,
            np.where(
                data[cols] > self.upper_bound,
                self.upper_bound,
                data[cols]
            )
           
    )
        return data
        
    # Denna metod skalar de angivna kolumnerna i data med hjälp av de skalare som har tränats på träningsdata.
    # StandardScaler används för 'age', 'thalach' och 'oldpeak_sqrt', medan RobustScaler används för 'trestbps' och 'chol' för att hantera eventuella outliers mer effektivt.
    # Metoden returnerar den transformerade data.
    def transform_data(self, data):
        colNames = ['age', 'thalach','oldpeak_sqrt']
        data[colNames] = self.scaler.transform(data[colNames])
        colNames = ['trestbps', 'chol']
        data[colNames] = self.robust_scaler.transform(data[colNames])
        return  data
    
    # Denna metod tränar skalare på de angivna kolumnerna i data. StandardScaler tränas på 'age', 'thalach' och 'oldpeak_sqrt', 
    # medan RobustScaler tränas på 'trestbps' och 'chol'. Detta görs för att förbereda skalare för att senare användas i transform_data-metoden.  
    def fit_data(self,data):
        colNames = ['age', 'thalach','oldpeak_sqrt']
        self.scaler.fit(data[colNames])
        colNames = ['trestbps', 'chol']
        self.robust_scaler.fit(data[colNames])
        return data

    def split_data(self, X, y):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return x_train, x_test, y_train, y_test
 
 
    def delete_duplicates(self, data):
        return data.drop_duplicates().reset_index(drop=True)    
# Denna metod kombinerar alla steg i dataförbehandlingen. Först hanterar den outliers i data genom att anropa handle_outliers-metoden med fit=True,
# sedan lägger den till en extra kolumn 'oldpeak_sqrt' genom att anropa add_extra_sqrt_column-metoden, tränar skalare på data genom att anropa fit_data-metoden,
# och slutligen transformerar data genom att anropa transform_data-metoden. Den returnerar den fullständigt transformerade data.
    def fit_transform(self, data):
        
        data = self.handle_outliers(data=data, fit=True)
        data = self.add_extra_sqrt_column(data=data)
        self.fit_data(data)
        data = self.transform_data(data)
        return data

    # Denna metod används för att transformera data utan att träna skalare eller beräkna outlier-gränser.
    # Den hanterar outliers i data genom att anropa handle_outliers-metoden med fit=False, 
    # lägger till en extra kolumn 'oldpeak_sqrt' genom att anropa add_extra_sqrt_column-metoden, och transformerar data genom att anropa transform_data-metoden. Den returnerar den transformerade data.   
    def transform(self, data):
       
        data = self.handle_outliers( data=data, fit=False)
        data = self.add_extra_sqrt_column(data=data)
        data = self.transform_data(data)
        return data
