# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from mlp_predict import SentimentModel, Sentiment_Predict

# Add title
st.title("AMAZON REVIEW SENTIMENT ANALYSIS SYSTEM")
# Add listbox in the sidebar 
choice = st.sidebar.selectbox("MENU", ("HOME", "ANALYSIS", "RESULTS"))

# Display HOME section
if choice == "HOME":
    st.image("https://miro.medium.com/v2/1*_JW1JaMpK_fVGld8pd1_JQ.gif")
    st.write("")
    st.markdown("##### • This system is a Natural Language Processing (NLP) application designed to analyze the sentiment of textual data.")
    st.markdown("##### • The application predicts sentiment in three categories: Positive, Negative, and Neutral.")
    st.markdown("##### • It also visualizes the results.")

# Display ANALYSIS section
elif choice == "ANALYSIS":
    gsid = st.text_input("Enter you Google Sheet ID")
    rng = st.text_input("Enter range between first column and last column")
    col = st.text_input("Enter the column name that is to be analyzed")
    btn = st.button("Analyze")
    if btn:
        # Build the Google Sheets API service and fetch data from the specified range in the Google Sheet
        if 'cred' not in st.session_state:
            f = InstalledAppFlow.from_client_secrets_file("key.json",["https://www.googleapis.com/auth/spreadsheets"])
            st.session_state['cred'] = f.run_local_server(port=0)
            service = build("Sheets", "v4", credentials=st.session_state['cred']).spreadsheets().values()
            d = service.get(spreadsheetId=gsid, range=rng).execute()
            data = d['values']
            df = pd.DataFrame(data=data[1:], columns=data[0])

            # Initialize the Sentiment predict model
            sentimentModel = Sentiment_Predict()
            l = []
            # Perform sentiment analysis on the specified column
            for i in range(len(df)):
                txt = df._get_value(i, col)
                pred = sentimentModel.predict_score(txt)
                if pred > 3:
                    l.append("Positive")
                elif pred < 3:
                    l.append("Negative")
                else:
                    l.append("Neutral")

            # Add the sentiment results to a new column in the DataFrame
            df["Sentiment"] = l
            df.to_csv("results.csv", index=False)
            st.subheader("The analysis results are saved by the name 'results.csv'")
# Display RESULTS section
elif choice=="RESULTS":
    # Read the results from the csv file
    df = pd.read_csv("results.csv")
    plot_choice = st.selectbox("Choose Visualization", ("NONE", "PIE CHART", "HISTOGRAM", "SCATTER PLOT"))
    st.dataframe(df)
    
    # Draw plots based on the users selection
    # Draw Pie chart
    if plot_choice == "PIE CHART":
        posper = (len(df[df['Sentiment']=="Positive"])/len(df)) * 100
        negper = (len(df[df['Sentiment']=="Negative"])/len(df)) * 100
        neuper = (len(df[df['Sentiment']=="Neutral"])/len(df)) * 100
        fig = px.pie(values=[posper, negper, neuper], names=['Positive', 'Negative', 'Neutral'])
        st.plotly_chart(fig)
    
    # Draw Histogram
    elif plot_choice == "HISTOGRAM":
        k = st.selectbox("Choose column", df.columns)
        if k:
            fig = px.histogram(x=df[k], color=df['Sentiment'])
            st.plotly_chart(fig)
    
    #Draw Scatter plot
    elif plot_choice == "SCATTER PLOT":
        k = st.text_input("Enter the continous column name")
        if k:
            fig = px.scatter(x=df[k], y=df['Sentiment'])
            st.plotly_chart(fig)
    
