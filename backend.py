import pandas as pd
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#Permission
f = InstalledAppFlow.from_client_secrets_file("key.json",["https://www.googleapis.com/auth/spreadsheets"])
cred = f.run_local_server(port=0)
service = build("Sheets", "v4", credentials=cred).spreadsheets().values()
d = service.get(spreadsheetId="1lQJpioXkiMm_5tUbszGs_Qc1TV95EM6nOz8gWI623Fo", range="A:C").execute()
data = d['values']
df = pd.DataFrame(data=data[1:], columns=data[0])
sentimentModel = SentimentIntensityAnalyzer()
for i in range(len(df)):
    txt = df._get_value(i, "Opinion")
#    pred = sentimentModel.polarity_scores(txt)
#    if pred['compound'] > 0.5:
#        data[i+1].append("Positive")
#    elif pred['compound'] < -0.5:
#        data[i+1].append("Negative")
#    else:
#        data[i+1].append("Neutral")

#h = {'values':data}
#service.update(spreadsheetId='1XhjsIQpwMcSbC1wh5EqoFs20S6pWwxgJFQicYH8PxdQ', 
#               range="B:G", valueInputOption="USER_ENTERED", body=h).execute()

