from django.shortcuts import render
from django.conf import settings
import joblib
import os
import pandas as pd

MODEL_PATH = os.path.join(settings.BASE_DIR, 'ML', 'mlmodel.pkl')
model = joblib.load(MODEL_PATH)

DATA_PATH = os.path.join(settings.BASE_DIR, 'notebooks', 'Datasets','train.csv')  # change to your actual CSV file
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['store', 'item', 'date'])

df['lag1'] = df.groupby(['store', 'item'])['sales'].shift(1)
df['lag7'] = df.groupby(['store', 'item'])['sales'].shift(7)
df['rolling7'] = df.groupby(['store', 'item'])['sales'].shift(1).rolling(window=7, min_periods=7).mean()
df.dropna(inplace=True)

def home(request):
    return render(request, 'home.html')

def fetch_features(store, item, target_date):
    historical = df[(df['store'] == store) &
                    (df['item'] == item) &
                    (df['date'] < target_date)].sort_values('date')
    
    if len(historical) < 7:
        return None, "Not enough historical data (need at least 7 days prior)."
    
    latest = historical.iloc[-1]
    lag1 = latest['sales']
    
    lag7 = historical.iloc[-8]['sales'] if len(historical) >= 8 else historical.iloc[0]['sales']
    
    rolling7 = historical.iloc[-7:]['sales'].mean()
    
    features = {
        'store': store,
        'item': item,
        'month': target_date.month,
        'week': target_date.isocalendar()[1],
        'day': target_date.day,
        'weekday': 1 if target_date.weekday() >= 5 else 0,
        'lag1': lag1,
        'lag7': lag7,
        'rolling7': rolling7,
    }
    return pd.DataFrame([features]), None

def result(request):
    context = {}
    if request.method == 'POST':
        try:
            store = int(request.POST.get('store'))
            item = int(request.POST.get('item'))
            date_str = request.POST.get('date')
            target_date = pd.to_datetime(date_str)
            
            input_data, error = fetch_features(store, item, target_date)
            
            if error:
                context['error'] = error
            else:
                pred = model.predict(input_data)[0]
                context['prediction'] = round(float(pred))
                context['date'] = target_date.strftime('%Y-%m-%d')
                context['item'] = item
                context['store'] = store
        except Exception as e:
            context['error'] = f"Invalid input: {str(e)}"
    
    return render(request, 'result.html', context)