# dashboard/app.py
import streamlit as st

st.set_page_config(page_title="KI-Trading-Agent", layout="wide")

st.title("📊 KI Trading Agent – Dashboard")
st.markdown("Hier wird dein Backtesting & Analyse-Interface entstehen.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Aktueller Status")
    st.metric("Balance", "10.000 €", "+3.5%")

with col2:
    st.subheader("Letztes Signal")
    st.write("Kaufe: NVIDIA – RSI 28 (überverkauft)")
