# from transformers import pipeline

# # Load the FinBERT model for sentiment analysis
# classifier = pipeline('sentiment-analysis',  model="ProsusAI/finbert", device=0) # device=0 for the first GPU

# # Example financial text
# text = "The company's earnings exceeded expectations, and the stock price surged."

# # Analyze sentiment
# result = classifier(text)
# print(result)







# GET ECONOMIC CALENDAR
# import investpy

# # Fetch economic calendar data
# calendar_data = investpy.economic_calendar(
#     from_date='01/12/2024',  # Start date (DD/MM/YYYY)
#     to_date='31/12/2024'    # End date (DD/MM/YYYY)
# )

# # # Display the first few rows
# # print(calendar_data.head())
# # Save to CSV
# calendar_data.to_csv('economic_calendar.csv', index=False)











# GET PRICES USD/CHF
# import yfinance as yf

# # # Fetch historical data for EUR/USD
# data = yf.download('USDCHF=X', interval='1h', period='1mo')  # Last 5 days of 15-minute data
# # 
# # Flatten MultiIndex columns
# data.columns = ['_'.join(col).strip() for col in data.columns.values]

# # Rename columns
# data.columns = ['Price Close', 'Price High', 'Price Low', 'Price Open', 'Volume']

# # Reset the index to make Datetime a regular column
# data.reset_index(inplace=True)

# # Rename the "Datetime" column if needed
# data.rename(columns={'index': 'Datetime'}, inplace=True)  # Optional if the index name isn't already 'Datetime'

# # print(data.head())
# data.to_csv('usd_chf_last_month.csv', index=False)







import pandas as pd

# Load data (replace with your actual files or dataframes)
forex_data = pd.read_csv('usd_chf_last_month.csv')
forex_data['timestamp'] = pd.to_datetime(forex_data['Datetime'])

# print(forex_data.head())

calendar_data = pd.read_csv('economic_calendar.csv')
calendar_data['time'] = calendar_data['time'].replace('All Day', '00:00')
calendar_data['time'] = calendar_data['time'].replace('ll DayA', '00:00')
calendar_data['time'] = calendar_data['time'].fillna('00:00')
calendar_data['timestamp'] = pd.to_datetime(calendar_data['date'] + ' ' + calendar_data['time'], errors='coerce', dayfirst=True)


# print(calendar_data.head())




forex_data = forex_data.sort_values(by='timestamp')
calendar_data = calendar_data.sort_values(by='timestamp')


# print(forex_data['timestamp'].dtype)  # Check type of the Forex data timestamps
# print(calendar_data['timestamp'].dtype)  # Check type of the Events data timestamps

forex_data['timestamp'] = forex_data['timestamp'].dt.tz_localize(None)
calendar_data['timestamp'] = calendar_data['timestamp'].dt.tz_localize(None)


# Merge the datasets based on the closest previous event timestamp
merged_data = pd.merge_asof(
    forex_data,          # Forex data
    calendar_data,         # Economic events data
    on='timestamp',      # Key column to merge on
    direction='backward' # Use the closest preceding event
)

# print(merged_data.columns.tolist())

# Filter rows where 'event' column is not null
merged_data = merged_data[merged_data['event'].notnull()]
merged_data.fillna({'actual': 0, 'forecast': 0, 'previous': 0}, inplace=True)

# # Lagged Prices: Add previous price values to capture trends.
merged_data['Close_Lag1'] = merged_data['Price Close'].shift(1)
merged_data['Close_Lag2'] = merged_data['Price Close'].shift(2)

# # Price Change: Calculate the price change between consecutive rows.
merged_data['price_change'] = merged_data['Price Close'] - merged_data['Close_Lag1']

# # Percentage Change: Calculate the percentage change in price.
merged_data['price_pct_change'] = (merged_data['price_change'] / merged_data['Close_Lag1']) * 100

merged_data['actual'] = pd.to_numeric(merged_data['actual'], errors='coerce')
merged_data['forecast'] = pd.to_numeric(merged_data['forecast'], errors='coerce')
merged_data['previous'] = pd.to_numeric(merged_data['previous'], errors='coerce')

merged_data.fillna({'actual': 0, 'forecast': 0, 'previous': 0}, inplace=True)

# Delta Features: Compute differences between actual and forecast values:
merged_data['delta_forecast'] = merged_data['actual'] - merged_data['forecast']
merged_data['delta_previous'] = merged_data['actual'] - merged_data['previous']

# # # Impact Score: Convert categorical impact values into numerical scores:
# impact_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
# merged_data['impact_score'] = merged_data['impact'].map(impact_mapping)

# print(merged_data.head())

# Get all unique values in the 'importance' column
# unique_importance_values = merged_data['importance'].unique()
# print(f"Unique values in 'importance': {unique_importance_values}")

# Fill missing values and reassign the column
merged_data['importance'] = merged_data['importance'].fillna('Unknown')

importance_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Unknown': 0}
merged_data['importance_score'] = merged_data['importance'].map(importance_mapping)
merged_data['Price_Direction'] = (merged_data['price_change'] > 0).astype(int)

# # Features
# X = merged_data[['Close_Lag1', 'Close_Lag2', 'importance_score', 'delta_forecast']].dropna()

# # # Target
# y = merged_data['Price_Direction']

# Drop rows with NaN values in X
X = merged_data[['Close_Lag1', 'Close_Lag2', 'importance_score', 'delta_forecast']].dropna()

# Align y with X's index
y = merged_data.loc[X.index, 'Price_Direction']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Shape of X: {X.shape}")
print(f"Length of y: {len(y)}")


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Detailed evaluation
print(classification_report(y_test, y_pred))




