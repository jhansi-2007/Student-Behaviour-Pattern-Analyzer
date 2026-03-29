import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("RCE Student Behaviour Pattern Analyzer")

st.write("Predict student dropout probability")

st.markdown(
"""
<style>
.stApp{
    background:linear-gradient(to right ,lightblue,lightgreen);
}
</style>
""",
unsafe_allow_html=True
)

# Sample training data
data = {
    "login_frequency": [45,40,35,30,25,20,15,10,5,2],
    "assignment_submission": [10,9,8,7,6,5,4,3,2,1],
    "marks": [90,85,80,75,70,65,60,55,45,35],
    "attendance": [95,90,85,80,75,70,65,60,55,50],
    "activities": [5,4,4,3,3,2,2,1,1,0],
    "gpa": [9.5,9.0,8.7,8.5,8.0,7.5,7.0,6.5,6.0,5.5],
    "dropout": [0,0,0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[[
    "login_frequency",
    "assignment_submission",
    "marks",
    "attendance",
    "activities",
    "gpa"
]]

y = df["dropout"]

model = LogisticRegression()
model.fit(X,y)

st.subheader("Enter Student Data")

# Student details
name = st.text_input("Student Name")
roll = st.text_input("Student Roll Number")

# Input features
login = st.number_input("Login Frequency (out of 50)",0,50)

assignment = st.number_input("Assignments Submitted (out of 10)",0,10)

marks = st.number_input("Marks Obtained (out of 100)",0,100)

attendance = st.number_input("Attendance Percentage",0,100)

activities = st.number_input(
    "Participation in Activities (Hackathons / Seminars / Workshops)",0,10
)

gpa = st.number_input(
    "Previous Semester GPA (0 - 10)",0.0,10.0,step=0.1
)

if st.button("Predict Dropout Risk"):

    prediction = model.predict([[login,assignment,marks,attendance,activities,gpa]])
    probability = model.predict_proba([[login,assignment,marks,attendance,activities,gpa]])

    risk = probability[0][1]

    st.write("Student Name:", name)
    st.write("Roll Number:", roll)

    st.write("Dropout Probability:", round(risk,2))

    if risk >= 0.6:
        st.error("High Dropout Risk")

    elif risk >= 0.3:
        st.warning("Medium Dropout Risk")

    else:
        st.success("Low Dropout Risk")