# Prediktion av hjärtsjukdom

## 📌 Beskrivning
Detta projekt är en AI-applikation utvecklad i Python för att förutsäga hjärtsjukdom baserat på medicinsk data. Modellen tränas på Heart Disease Dataset och användaren kan mata in värden via ett Tkinter-gränssnitt för att få en prediktion.

---

## 📊 Dataset
Datasetet som används är Heart Disease Dataset från Kaggle.  
Det innehåller medicinska variabler såsom:
- Ålder (age)
- Kön (sex)
- Bröstsmärta (chest pain type)
- Blodtryck (resting blood pressure)
- Kolesterol (cholesterol)
- Maxpuls (maximum heart rate)
- Thalassemia (thal)
- med mera

Målvariabel:
- 0 = Ingen hjärtsjukdom  
- 1 = Hjärtsjukdom  

---

## ⚙️ Tekniker
- Python  
- pandas  
- numpy  
- matplotlib / seaborn  
- scikit-learn  
- xgboost  
- tkinter  

---

## 🤖 Modeller
Följande maskininlärningsmodeller har tränats och utvärderats:
- Logistic Regression  
- Random Forest  
- XGBoost  

### Resultat:

| Modell               | Accuracy | Precision | Recall | F1-score |
|--------------------|----------|-----------|--------|----------|
| XGBoost            | 0.79     | 0.81      | 0.79   | 0.80     |
| Logistic Regression| 0.82     | 0.84      | 0.82   | 0.83     |
| Random Forest      | 0.79     | 0.79      | 0.82   | 0.81     |

Logistic Regression presterade bäst och valdes som huvudmodell i applikationen.

---

## 🖥️ Applikation
En applikation byggd med Tkinter har utvecklats där användaren kan:
- Mata in medicinska värden  
- Validera input  
- Få prediktion från modellen  

---

## 📁 Projektstruktur
