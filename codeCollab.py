import pandas as pd
from textblob import TextBlob

# read the CSV file into a pandas DataFrame
df = pd.read_csv('messages.csv')

# create empty lists to store polarity and subjectivity values
polarity = []
subjectivity = []

# iterate over each row in the DataFrame
for index, row in df.iterrows():
    # create a TextBlob object for the text in the current row
    blob = TextBlob(row['text'])
    # append the polarity and subjectivity values to the lists
    polarity.append(blob.sentiment.polarity)
    subjectivity.append(blob.sentiment.subjectivity)

# add the polarity and subjectivity columns to the DataFrame
df['polarity'] = polarity
df['subjectivity'] = subjectivity

# save the updated DataFrame to a new CSV file with headers 'text', 'polarity', and 'subjectivity'
df.to_csv('data.csv', index=False, columns=['text', 'polarity', 'subjectivity','scam'])

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def predict_average_polarity(csv_file):
    # Load the data from the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_file)

    # Preprocess the data
    X = data[['scam']]
    y = data['polarity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a machine learning model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Predict the average polarity for scam messages
    y_pred = model.predict(X_test[X_test['scam'] == 1])
    average_polarity = y_pred.mean()

    return average_polarity

average_polarity = predict_average_polarity("data.csv")
print("Average Polarity for Scam Messages:", average_polarity)
