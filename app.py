
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# Load dataset
DATA_PATH = "final_processed_fraud_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Ensure 'Company_Name' column exists
if 'Company_Name' not in df.columns:
    st.error("Dataset must contain 'Company_Name' column.")
    st.stop()

feature_cols = ['Delay_Hours', 'Route_Deviation', 'Invoice_Mismatch_Flag', 'Warehouse_Mismatch_Flag', 'Complaint_Count']

# Preprocess data
X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
klabels = kmeans.fit_predict(X_scaled)

# Mapping dummy true labels (for demonstration)
def best_map(y_true, y_pred):
    labels_pred = np.unique(y_pred)
    pred_map = {v:i for i,v in enumerate(labels_pred)}
    y_pred_mapped = np.array([pred_map[v] for v in y_pred])
    D = max(y_pred_mapped.max(), y_true.max()) + 1
    cost = np.zeros((D, D), dtype=int)
    for i in range(len(y_true)):
        cost[y_pred_mapped[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(cost.max() - cost)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[labels_pred[r]] = c
    y_remap = np.array([mapping.get(v, 999) for v in y_pred])
    return y_remap

mapped_labels = best_map(np.zeros(len(klabels)), klabels)

df['Cluster_Label'] = mapped_labels

# Streamlit Interface
st.title("🚚 Fraud Shipment Identification System")

option = st.sidebar.selectbox("Choose an Action", ["Check Company", "Search by Factor"])

if option == "Check Company":
    company_name = st.text_input("Enter Company Name")
    if st.button("Check Status"):
        if company_name in df['Company_Name'].values:
            row = df[df['Company_Name'] == company_name].iloc[0]
            status = 'Trusted' if row['Cluster_Label'] != 2 else 'Not Trusted'
            st.success(f"Company: {row['Company_Name']}")
            st.write(f"✅ Authorized Status: **{status}**")
            st.write(f"- Route Deviations: {row['Route_Deviation']}")
            st.write(f"- Invoice Mismatches: {row['Invoice_Mismatch_Flag']}")
            st.write(f"- Timestamp Mismatches: {row['Warehouse_Mismatch_Flag']}")
            st.write(f"- Complaint Count: {row['Complaint_Count']}")
        else:
            st.error("Company not found in dataset.")

elif option == "Search by Factor":
    factor = st.selectbox("Select Factor", feature_cols)
    if st.button("Show Companies with Issue"):
        filtered = df[df[factor] > 0][['Company_Name', factor]]
        if not filtered.empty:
            st.write(f"📋 Companies with {factor} issues:")
            st.dataframe(filtered.reset_index(drop=True))
        else:
            st.info("No companies found with this issue.")
