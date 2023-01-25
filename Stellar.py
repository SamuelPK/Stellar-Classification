import streamlit as st
import pickle
import numpy as np, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
loaded_rf = pickle.load(open('Model.pkl', "rb"))


st.title("Stellar Analysis")
st.subheader("Context")
st.write("""
In astronomy, stellar classification is the classification of stars based on their spectral characteristics. The classification scheme of galaxies, quasars, and stars is one of the most fundamental in astronomy. The early cataloguing of stars and their distribution in the sky has led to the understanding that they make up our own galaxy and, following the distinction that Andromeda was a separate galaxy to our own, numerous galaxies began to be surveyed as more powerful telescopes were built. This datasat aims to classificate stars, galaxies, and quasars based on their spectral characteristics.

The data consists of 100,000 observations of space taken by the SDSS (Sloan Digital Sky Survey). Every observation is described by 17 feature columns and 1 class column which identifies it to be either a star, galaxy or quasar.
""")

st.header("Stellar Dataset")

#load the data
df = pd.read_csv("./star_classification.csv")
#Feature Enginnering
df.drop(['obj_ID','run_ID','rerun_ID','cam_col','field_ID', 'spec_obj_ID','fiber_ID'], axis = 1, inplace=True)
df["class"] = df["class"].map({"GALAXY":0,"STAR":1,"QSO":2})
outliers = df.quantile(.97)
df = df[(df['redshift']<outliers['redshift'])]
df = df[(df['i']<outliers['i'])]
df = df[(df['plate']<outliers['plate'])]

# assign x and y
#undersampling
x = df.drop(['class'], axis = 1)
y = df.loc[:,'class'].values
us = RandomUnderSampler()
x, y = us.fit_resample(x, y)


# Correlation heatmap for picture purposes
fig1 = plt.figure(figsize=(17,8))
sns.heatmap(df.corr(),annot=True);
st.pyplot(fig1)
st.write("This is the correlation heatmap after cleaning data")


# Scale the Data - Data Standartization
from sklearn.preprocessing import StandardScaler
ssc = StandardScaler()
scale = ssc.fit_transform(x) # use standard scaler to scale the data

st.header("Use this dataframe for reference values")
st.write(df.describe())     
st.success("Enter the measured data: ")
with st.form("predict_form"):
    
    input1 = st.text_input("Alpha (0,360)")
    input2 = st.text_input('Delta (-18,63)')
    input3 = st.text_input('UV value (0, 30)')
    input4 = st.text_input('Green value (0, 32)')
    input5 = st.text_input('Red value (9, 30)')
    input6 = st.text_input('Near-infrared value(i) (9, 24)')
    input7 = st.text_input('Infrared value(z) (0, 30)')
    input8 = st.text_input('Redshift value (0, 2.5)')
    input9 = st.text_input('Plate ID (266, 11000)')
    input10 = st.text_input('MJ date (51000, 59000)')

    unseen_data=pd.DataFrame({
                            'Alpha':[input1],
                            'Delta':[input2],
                            'UV value':[input3],
                            'Green value':[input4],
                            'Red value':[input5],
                            'Near-infrared value(i)':[input6],
                            'Infrared value(z)':[input7],
                            'Redshift value':[input8],
                            'Plate ID':[input9],
                            'MJ date':[input10],
                            })
    submitted = st.form_submit_button("Predict")
    if submitted:
        loaded_rf = pickle.load(open('Model.pkl', "rb"))
        result = loaded_rf.predict(unseen_data)
        st.write(result)

st.warning('"GALAXY":0,"STAR":1,"QSO":2')
