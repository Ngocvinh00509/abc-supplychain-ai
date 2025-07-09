import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 🚨 Main function to perform SKU Risk Classification using Random Forest
def render(df):
    st.header("🔍 SKU Risk Classification")

    # ✅ Step 1: Check required columns
    required_cols = ['stock', 'quantity', 'sku']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"❌ Dataset must include a '{col}' column to classify SKU risk.")
            return

    df_risk = df.copy()

    # ✅ Step 2: Handle missing values for key numerical columns
    df_risk['stock'] = df_risk['stock'].fillna(0)
    df_risk['quantity'] = df_risk['quantity'].fillna(0)

    # ✅ Step 3: Feature Engineering – Calculate Sell-Through Rate
    df_risk['sell_through_rate'] = df_risk['quantity'] / (df_risk['stock'] + 1)

    # ✅ Step 4: Risk Classification based on Sell-Through Rate
    df_risk['sku_status'] = pd.cut(
        df_risk['sell_through_rate'],
        bins=[-1, 0.3, 0.7, float('inf')],
        labels=['Overstock', 'Balanced', 'Understock']
    )

    # ✅ Step 5: Encode categorical features
    for col in ['region', 'event', 'sku']:
        if col in df_risk.columns:
            df_risk[col] = df_risk[col].astype(str)
        else:
            df_risk[col] = 'Unknown'

    df_risk['region_enc'] = LabelEncoder().fit_transform(df_risk['region'])
    df_risk['event_enc'] = LabelEncoder().fit_transform(df_risk['event'])
    df_risk['sku_enc'] = LabelEncoder().fit_transform(df_risk['sku'])

    # ✅ Step 6: Prepare features and labels
    features = ['stock', 'quantity', 'region_enc', 'event_enc', 'sku_enc']
    X = df_risk[features]
    y = df_risk['sku_status']

    # ✅ Step 7: Train/Test Split & Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # ✅ Step 8: Model Evaluation
    y_pred = clf.predict(X_test)
    eval_result = {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()  # For display compatibility
    }

    # ✅ Step 9: Predict entire dataset and attach results
    df_risk['predicted_risk'] = clf.predict(X)

    # Step 10: Display Results
    st.success("✅ SKU Risk Classification Completed!")
    st.dataframe(df_risk[['sku', 'stock', 'quantity', 'sell_through_rate', 'predicted_risk']].sort_values(by='predicted_risk'))

    # Pie chart: Risk distribution
    fig_pie = px.pie(df_risk, names='predicted_risk', title='🔍 Distribution of SKU Risk')
    st.plotly_chart(fig_pie, use_container_width=True)

    # Bar chart: Top 10 understock SKUs
    understock_df = df_risk[df_risk['predicted_risk'] == 'Understock']
    if not understock_df.empty:
        fig_top_under = px.bar(
            understock_df.nlargest(10, 'sell_through_rate'),
            x='sku', y='sell_through_rate',
            title='🔥 Top 10 Understock SKUs'
        )
        st.plotly_chart(fig_top_under, use_container_width=True)
    else:
        st.info("ℹ No SKUs classified as 'Understock'.")

    # ✅ Evaluation Metrics Summary
    st.subheader("📊 Model Evaluation")
    st.write(f"**Accuracy:** {eval_result['accuracy']:.2f}")
    st.text("Classification Report:")
    st.json(eval_result['report'])
    st.text("Confusion Matrix:")
    st.write(eval_result['confusion_matrix'])
