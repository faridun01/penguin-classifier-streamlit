import streamlit as st
import pandas as pd

st.title('🎈 App Name')

st.write('Hello world!')

with st.expander("DATA"):
  st.write('**OUR DATA**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df

