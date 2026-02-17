import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from skfuzzy import control as ctrl
import skfuzzy as fuzz


df = pd.read_csv('large_crop_rotation_data.csv')  


print(df.head())

df['previous_crop'] = pd.Categorical(df['previous_crop']).codes  
df['recommended_next_crop'] = pd.Categorical(df['recommended_next_crop']).codes  


soil_nitrogen = ctrl.Antecedent(np.arange(0, 51, 1), 'soil_nitrogen')
soil_ph = ctrl.Antecedent(np.arange(5.0, 8.1, 0.1), 'soil_ph')
rainfall = ctrl.Antecedent(np.arange(0, 2001, 1), 'rainfall')
temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
crop = ctrl.Consequent(np.arange(0, 6, 1), 'crop')


soil_nitrogen['low'] = fuzz.trimf(soil_nitrogen.universe, [0, 0, 25])
soil_nitrogen['medium'] = fuzz.trimf(soil_nitrogen.universe, [0, 25, 50])
soil_nitrogen['high'] = fuzz.trimf(soil_nitrogen.universe, [25, 50, 50])

soil_ph['acidic'] = fuzz.trimf(soil_ph.universe, [5.0, 5.0, 6.5])
soil_ph['neutral'] = fuzz.trimf(soil_ph.universe, [5.5, 6.5, 7.5])
soil_ph['alkaline'] = fuzz.trimf(soil_ph.universe, [6.5, 7.5, 8.0])

rainfall['low'] = fuzz.trimf(rainfall.universe, [0, 0, 1000])
rainfall['medium'] = fuzz.trimf(rainfall.universe, [0, 1000, 2000])
rainfall['high'] = fuzz.trimf(rainfall.universe, [1000, 2000, 2000])

temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 20])
temperature['medium'] = fuzz.trimf(temperature.universe, [10, 20, 30])
temperature['high'] = fuzz.trimf(temperature.universe, [20, 30, 40])

crop['barley'] = fuzz.trimf(crop.universe, [0, 0, 1])
crop['cotton'] = fuzz.trimf(crop.universe, [0, 1, 2])
crop['maize'] = fuzz.trimf(crop.universe, [1, 2, 3])
crop['rice'] = fuzz.trimf(crop.universe, [2, 3, 4])
crop['soybean'] = fuzz.trimf(crop.universe, [3, 4, 5])
crop['tomato'] = fuzz.trimf(crop.universe, [4, 5, 6])


rule1 = ctrl.Rule(soil_nitrogen['high'] & soil_ph['neutral'] & rainfall['medium'] & temperature['medium'], crop['maize'])
rule2 = ctrl.Rule(soil_nitrogen['medium'] & soil_ph['neutral'] & rainfall['high'] & temperature['high'], crop['cotton'])
rule3 = ctrl.Rule(soil_nitrogen['low'] & soil_ph['acidic'] & rainfall['low'] & temperature['low'], crop['barley'])


crop_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
crop_sim = ctrl.ControlSystemSimulation(crop_ctrl)


X = df[['soil_nitrogen', 'soil_ph', 'rainfall', 'temperature', 'previous_crop']]  # Features
y = df['recommended_next_crop']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)


plt.figure(figsize=(10, 5))
sns.barplot(x=rf_model.feature_importances_, y=X.columns)
plt.title("Feature Importance")
plt.show()





def recommend_crop(soil_nitrogen_input, soil_ph_input, rainfall_input, temperature_input, previous_crop_input):
    # Pass user inputs to the fuzzy system
    crop_sim.input['soil_nitrogen'] = soil_nitrogen_input
    crop_sim.input['soil_ph'] = soil_ph_input
    crop_sim.input['rainfall'] = rainfall_input
    crop_sim.input['temperature'] = temperature_input

  
    crop_sim.compute()

   
    recommended_crop = crop_sim.output['crop']
    
    
    crop_mapping = {
        0: 'Barley', 1: 'Cotton', 2: 'Maize', 3: 'Rice', 4: 'Soybean', 5: 'Tomato'
    }
    
    return crop_mapping.get(int(recommended_crop), 'No recommendation')


from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.fit(['Soybean', 'Millet', 'Maize', 'Cotton', 'Barley', 'Potato', 'Rice'])

def rf_predict(soil_nitrogen, soil_ph, rainfall, temperature, previous_crop):
    try:
        previous_crop_encoded = label_encoder.transform([previous_crop])
    except ValueError:
        print(f"Error: Unseen label '{previous_crop}' encountered.")
        return 'No recommendation'  # Return a default value or handle accordingly

    features = [soil_nitrogen, soil_ph, rainfall, temperature, previous_crop_encoded[0]]

    rf_prediction = rf_model.predict([features])[0]

    crop_mapping = {
        0: 'Barley', 1: 'Cotton', 2: 'Maize', 3: 'Rice', 4: 'Soybean', 5: 'Tomato'
    }
    predicted_crop = crop_mapping.get(rf_prediction, 'No recommendation')

    return predicted_crop






soil_nitrogen_input = float(input("Enter soil nitrogen level (0-50): "))
soil_ph_input = float(input("Enter soil pH (5.0-8.0): "))
rainfall_input = float(input("Enter rainfall (0-2000 mm): "))
temperature_input = float(input("Enter temperature (0-40°C): "))
previous_crop_input = input("Enter previous crop (Soybean, Millet, Maize, Cotton, Barley, Potato, Rice): ")

fuzzy_crop = recommend_crop(soil_nitrogen_input, soil_ph_input, rainfall_input, temperature_input, previous_crop_input)
rf_crop = rf_predict(soil_nitrogen_input, soil_ph_input, rainfall_input, temperature_input, previous_crop_input)
recommended_crop = crop_sim.output['crop']
print(f"Fuzzy Logic Recommended Crop: {fuzzy_crop}")
rf_crop = rf_predict(25, 6.5, 1200, 25, 'Maize')
print(f"Random Forest Recommended Crop: {rf_crop}")


crops = ['Fuzzy Logic', 'Random Forest']
predictions = [fuzzy_crop, rf_crop]

plt.figure(figsize=(10, 5))
sns.barplot(x=crops, y=predictions, palette='Set1')
plt.title('Crop Prediction Comparison: Fuzzy Logic vs Random Forest')
plt.ylabel('Predicted Crop')
plt.show()



plt.figure(figsize=(10, 8))
correlation_matrix = df[['soil_nitrogen', 'soil_ph', 'rainfall', 'temperature', 'previous_crop']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()
