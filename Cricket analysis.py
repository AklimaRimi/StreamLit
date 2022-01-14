import streamlit as st
import pandas as pd
#"E:\research paper\t20_matches.csv"
#data = pd.read_csv("E:/research paper/t20_matches.csv")

st.write("""
# Cricket analysis
* **Libraries:** Streamlit, pandas, Matplotlib, Seaborn
* **Date:** 27/11/2021
* **Editor:** Aklima Akter Rimi

# Filtered Dataframe
""")
@st.cache
def loading_data():
    data = pd.read_csv("E:/research paper/IPL Matches 2008-2020.csv")
    data = data.drop(columns = ['city','id','venue','neutral_venue','eliminator','method'])
    data = data.fillna(method='ffill')
    return data
data = loading_data()

w_names = data['winner'].unique()
select_winner = st.sidebar.multiselect('winner',w_names,w_names)

p_names = data['player_of_match'].unique()
select_players = st.sidebar.multiselect('Player of the match',p_names,p_names)

t_names = data['toss_decision'].unique()
select_toss = st.sidebar.multiselect('Toss Decision',t_names,t_names)



data = data[(data['winner'].isin(select_winner)) & (data['player_of_match'].isin(select_players)) & (data['toss_decision'].isin(select_toss))]

st.dataframe(data)
"There are "+ str(data.shape[0]) + ' informations.'