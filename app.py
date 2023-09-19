import streamlit as st
import pickle
import pandas as pd

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bengaluru', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali']

pipe = pickle.load(open('pipe.pkl','rb'))

st.title("IPL Win Probability Predictor")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the team batting first', sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the team bowling first', sorted(teams))

city = st.selectbox('Select the match city', sorted(cities))

target = st.number_input('Enter the Target Runs', value=0, step=5, format = "%d")

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score', value=0, step=5, format = "%d")

with col4:
    overs = st.number_input('Overs Completed', value=0, step=1, format = "%d", max_value = 20)

with col5:
    wickets = st.number_input('Wickets Out', value=0, step=1, format = "%d", max_value = 10,)

if st.button('Predict Win Probability'):
    runs_left = target - score
    balls_left = 120 - overs*6
    wickets_left = 10 - wickets
    CRR = score/overs
    RRR = runs_left*6/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],
                             'city':[city],'runs_left':[runs_left],'balls_left':[balls_left],
                             'wickets_left':[wickets_left],'total_runs_x':[target],'CRR':[CRR],'RRR':[RRR]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + " - " + str(round(win*100)) + "%")
    st.header(bowling_team + " - " + str(round(loss*100)) + "%")

