import streamlit as st
from evaluate_standard import evaluate_all
import pandas as pd
import matplotlib.pyplot as plt

st.title("Cranfield Standard Evaluation — Boolean vs LTR")

k = st.slider("Select cutoff k", 5, 30, 10)
if st.button("Run Full Evaluation"):
    with st.spinner("Evaluating all 225 Cranfield queries..."):
        results = evaluate_all(k=k)

    df = pd.DataFrame([
        {"Metric": m, "Boolean": v[0], "LTR": v[1]} for m, v in results.items()
    ])
    st.dataframe(df.style.format("{:.4f}"))

    fig, ax = plt.subplots()
    df.set_index("Metric")[["Boolean", "LTR"]].plot(kind="bar", ax=ax)
    plt.title(f"Performance Comparison (k={k})")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    st.pyplot(fig)
