# 📁 FILE: models/rf_classifier_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_rf_model(df: pd.DataFrame):
    df = df.copy()

 
    df['sell_through_rate'] = df['quantity'] / (df['stock'] + 1)
    
    df['sku_status'] = pd.cut(df['sell_through_rate'],
        bins=[-1, 0.3, 0.7, float('inf')],
        labels=['Overstock', 'Balanced', 'Understock'])

    for col in ['region', 'event', 'sku']:
        if col in df.columns:
            df[col] = df[col].astype(str)
        else:
            df[col] = 'Unknown'

    le_region = LabelEncoder()
    le_event = LabelEncoder()
    le_sku = LabelEncoder()

    df['region_enc'] = le_region.fit_transform(df['region'])
    df['event_enc'] = le_event.fit_transform(df['event'])
    df['sku_enc'] = le_sku.fit_transform(df['sku'])

    features = ['stock', 'quantity', 'region_enc', 'event_enc', 'sku_enc']
    X = df[features]
    y = df['sku_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # 🎯 Dự đoán trên tập test để đánh giá
    y_pred = clf.predict(X_test)

    # 📊 Tính toán các chỉ số đánh giá
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    evaluation = {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": conf_matrix
    }

    return clf, evaluation

