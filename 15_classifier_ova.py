''' 
Command: python3 15_classifier_ova.py <filename>
Example: python3 15_classifier_ova.py penguins_test.csv
'''


# Import libraries
import pandas as pd       # Importing for panel data analysis
import sys
import pickle             # Loading model

# print('Number of arguments:', len(sys.argv), 'arguments.')

''' Let's analyze the data in **penguins_test.csv** '''
# Import training data into pandas dataframe
if (len(sys.argv) == 1):
    print("Filename argument missing. Taking the default file \"penguins_test.csv\" for making predictions")
    filename = './penguins_test.csv'
else:
    filename = str(sys.argv[1])
test_csv = pd.read_csv(filename)
# print('Shape of the dataset:', test_csv.shape)
# test_csv.head(3)

# Let's drop the null values, as to make predictions using the trained SVM ono-vs-all and one-vs-one classifiers, we need the values of all the attributes.
test_csv.dropna(inplace=True)

X_encoded = test_csv.copy()

''' Let's normalize the data (continuous attributes). We are choosing z-score normalization to retain the outlier information. '''
X_encoded["Culmen Length (mm)"] = (X_encoded["Culmen Length (mm)"] - X_encoded["Culmen Length (mm)"].mean())/X_encoded["Culmen Length (mm)"].std()
X_encoded["Culmen Depth (mm)"] = (X_encoded["Culmen Depth (mm)"] - X_encoded["Culmen Depth (mm)"].mean())/X_encoded["Culmen Depth (mm)"].std()
X_encoded["Flipper Length (mm)"] = (X_encoded["Flipper Length (mm)"] - X_encoded["Flipper Length (mm)"].mean())/X_encoded["Flipper Length (mm)"].std()
X_encoded["Body Mass (g)"] = (X_encoded["Body Mass (g)"] - X_encoded["Body Mass (g)"].mean())/X_encoded["Body Mass (g)"].std()
X_encoded["Delta 15 N (o/oo)"] = (X_encoded["Delta 15 N (o/oo)"] - X_encoded["Delta 15 N (o/oo)"].mean())/X_encoded["Delta 15 N (o/oo)"].std()
X_encoded["Delta 13 C (o/oo)"] = (X_encoded["Delta 13 C (o/oo)"] - X_encoded["Delta 13 C (o/oo)"].mean())/X_encoded["Delta 13 C (o/oo)"].std()

# test_csv.head(3)

''' Let's preprocess this **test_csv** the same way as we did for train_csv '''
# Prepare data. This format must match the `X_train_ova` or `X_train_ovo` from `15_Assignment4.ipynb`
data = X_encoded[['Island', 'Clutch Completion', 'Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Sex', 'Delta 15 N (o/oo)', 'Delta 13 C (o/oo)']]
X_encoded = pd.get_dummies(data, columns=['Island'])
X_encoded.replace({'Yes': 1, 'No': 0}, inplace=True)
X_encoded.replace({'FEMALE': 1, 'MALE': 0}, inplace=True)
X_encoded.replace({True: 1, False: 0}, inplace=True)

svm_classifiers_ova = {} # Initialize a dictionary to store classifiers for each of the three binary class problems

# load the model from disk
loaded_model_adelie = pickle.load(open('./models/ova_svm_final_adelie.sav', 'rb'))
svm_classifiers_ova['Adelie Penguin (Pygoscelis adeliae)'] = loaded_model_adelie
loaded_model_gentoo = pickle.load(open('./models/ova_svm_final_gentoo.sav', 'rb'))
svm_classifiers_ova['Gentoo penguin (Pygoscelis papua)'] = loaded_model_gentoo
loaded_model_chinstrap = pickle.load(open('./models/ova_svm_final_chinstrap.sav', 'rb'))
svm_classifiers_ova['Chinstrap penguin (Pygoscelis antarctica)'] = loaded_model_chinstrap
# print(svm_classifiers_ova)
# it should be of the form:
# svm_classifiers_ova {'Adelie Penguin (Pygoscelis adeliae)': SVC(C=0.006, kernel='linear', random_state=42), 'Gentoo penguin (Pygoscelis papua)': SVC(C=0.006, kernel='linear', random_state=42), 'Chinstrap penguin (Pygoscelis antarctica)': SVC(C=0.006, kernel='linear', random_state=42)}

''' Making predictions on Test set via trained binary classifiers '''
per_class_pred_ova = {} # Initialize a dictionary to store prediction class

for class_, svm_classifier in svm_classifiers_ova.items():
    # Predict using the trained SVM classifier
    y_pred_ova = svm_classifier.predict(X_encoded)
    # print(f"For binary classifier with class - {class_}, \nthe predicted output for each input tuple is: {y_pred_ova}\n")
    per_class_pred_ova[class_] = y_pred_ova

# print("The final dictionary comprising the predicitions on input tuples from each of the three binary classifiers look like:")
# print(per_class_pred_ova)

# Combine the predictions to get multi-class predictions
y_pred_multiclass_ova = pd.DataFrame(per_class_pred_ova).idxmax(axis=1)
print(y_pred_multiclass_ova)

''' Save the data to a csv file called `ova.csv`'''
# Appending the prediction results in the dataframe
test_csv['predicted'] = y_pred_multiclass_ova.values
test_csv.to_csv('ova.csv', index=False)

print("*"*10)
print("*"*10)
print("Successfully created the file ova.csv")