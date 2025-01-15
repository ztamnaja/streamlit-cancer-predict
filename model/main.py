import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle 

pd.set_option('display.max_columns', None)
def get_clean_data():
    df = pd.read_csv('./dataset/data.csv')

    df.drop(columns=['Unnamed: 32', 'id'], axis=1,inplace=True)

    df['diagnosis'] = [1 if value == "M" else 0 for value in df['diagnosis']]
    df['diagnosis'] = df['diagnosis'].astype("category", copy=False)
    return df


def create_model(df):
    scaler = StandardScaler()
    lr = LogisticRegression()

    y = df['diagnosis']
    X = df.drop(columns=['diagnosis'], axis=1)

    x_scaled = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    lr.fit(x_train, y_train)
    
    y_pred = lr.predict(x_test)

    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return lr, scaler


def main():
    
    data = get_clean_data()

    model, scaler = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()