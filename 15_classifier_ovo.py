''' 
Command: python3 15_classifier_ovo.py <filename>
Example: python3 15_classifier_ovo.py penguins_test.csv
'''


# Import libraries
import pandas as pd       # Importing for panel data analysis
import numpy as np
import pickle             # Loading model
import sys
import warnings           # Importing warning to disable runtime warnings
warnings.filterwarnings("ignore") 


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

svm_classifiers_ovo = {} # Initialize a dictionary to store classifiers for each of the three binary class problems

# load the model from disk
loaded_model_adelie_gentoo = pickle.load(open('./models/ovo_svm_final_adelie_gentoo.sav', 'rb'))
svm_classifiers_ovo[('Adelie Penguin (Pygoscelis adeliae)', 'Gentoo penguin (Pygoscelis papua)')] = loaded_model_adelie_gentoo
loaded_model_adelie_chinstrap = pickle.load(open('./models/ovo_svm_final_adelie_chinstrap.sav', 'rb'))
svm_classifiers_ovo[('Adelie Penguin (Pygoscelis adeliae)', 'Chinstrap penguin (Pygoscelis antarctica)')] = loaded_model_adelie_chinstrap
loaded_model_gentoo_chinstrap = pickle.load(open('./models/ovo_svm_final_gentoo_chinstrap.sav', 'rb'))
svm_classifiers_ovo[('Gentoo penguin (Pygoscelis papua)', 'Chinstrap penguin (Pygoscelis antarctica)')] = loaded_model_gentoo_chinstrap
# print(svm_classifiers_ovo)
# it should be of the form:
# svm_classifiers_ovo  {('Adelie Penguin (Pygoscelis adeliae)', 'Gentoo penguin (Pygoscelis papua)'): SVC(C=0.005, kernel='linear', random_state=42), ('Adelie Penguin (Pygoscelis adeliae)', 'Chinstrap penguin (Pygoscelis antarctica)'): SVC(C=0.005, kernel='linear', random_state=42), ('Gentoo penguin (Pygoscelis papua)', 'Chinstrap penguin (Pygoscelis antarctica)'): SVC(C=0.005, kernel='linear', random_state=42)}

''' Making predictions on Test set via trained binary classifiers '''
predictions_ovo = [] # Initialize an empty list to store predictions

# Iterate through each data point in the test set
for index, row in X_encoded.iterrows():
    # print("row:\n", row)
    class_votes = {} # Initialize a dictionary to store class votes
    
    # Use each OvO classifier to vote for a class
    for (class_1, class_2), svm_classifier in svm_classifiers_ovo.items():
        y_pred = svm_classifier.predict([row])[0]
    #     # print(f"For classes {class_1} & {class_2}")
    #     # print("y_pred: ", y_pred)
        if y_pred == 1:
            class_votes[class_1] = class_votes.get(class_1, 0) + 1
        else:
            class_votes[class_2] = class_votes.get(class_2, 0) + 1
    #     print("class_votes: ", class_votes)
    
    # Determine the predicted class with the most votes
    predicted_class = max(class_votes, key=class_votes.get)
#     # print("predicted_class of the row: ", predicted_class)

    predictions_ovo.append(predicted_class)
print("predictions for all Test rows: ", predictions_ovo)

''' Save the data to a csv file called `ova.csv`'''
# Appending the prediction results in the dataframe
test_csv['predicted'] = np.array(predictions_ovo)
test_csv.to_csv('ovo.csv', index=False)

print("*"*10)
print("*"*10)
print("Successfully created the file ovo.csv")