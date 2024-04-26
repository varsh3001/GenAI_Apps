import streamlit as st
from logic import rag_system
st.title("RAG System on 'Leave No Context Behind' Paper")

# Customizing Streamlit's background color
bg = """
<style>
[data-testid="stAppContainer"] {
  background-color: #76D7EA;
}
</style>
"""
st.markdown(bg, unsafe_allow_html=True)

label = "Enter the query regarding the paper"
query = st.text_input(label)

if st.button("Generate Answer"):
  res=rag_system(query)
  st.markdown(res)



     