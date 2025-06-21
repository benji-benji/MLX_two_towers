# streamlit_app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000/search"

st.title("🔍 Two-Tower Search")

query = st.text_input("Enter your search query:")

if st.button("Search") and query.strip():
    with st.spinner("Searching..."):
        response = requests.get(API_URL, params={"q": query})
        if response.status_code == 200:
            results = response.json()
            if results:
                for res in results:
                    st.markdown(f"**Score:** {res['score']:.4f}")
                    st.markdown(f"**Doc:** {res['doc']}")
                    st.markdown("---")
            else:
                st.info("No results found.")
        else:
            st.error("Error querying the search API.")
