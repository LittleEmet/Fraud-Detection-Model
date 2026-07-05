# uvicorn script1:app --host 0.0.0.0 --port 8000!copy this to run the code!
import pandas as pd
from preprocess import clean1, encode1
from datetime import date
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import torch
from pyngrok import ngrok
from  Model import Net
model = Net()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()
encoder = joblib.load("onehot_encoder.joblib")
scaler = joblib.load("scaler.pkl")
mapping = {
    "nofraud": 0,
    "Phishing": 1,
    "Account Takeover": 2,
    "Synthetic Identity": 3,
    "Card Cloning": 4,
    "Friendly Fraud": 5,
    'Identity Theft': 6
}
idx_to_label = {v: k for k, v in mapping.items()}
app = FastAPI()
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """<h4>This is a fraud detector model!</h4><a href='/input'><button>Input your data?</button></a>"""
@app.get("/input", response_class=HTMLResponse)
def read_input():
    return """<h1>Please input your data!</h1>
    <form action="/predict" method="post">
    <form>

  <!-- BASIC TRANSACTION TIME INPUTS -->
  <label>Transaction Date</label>
  <input type="date" name="transaction_date"><br>

  <label>Transaction Time</label>
  <input type="time" name="transaction_time"><br>


  <!-- NUMERIC FEATURES -->
  <label>Customer Age</label>
  <input type="number" name="customer_age"><br>

  <label>Credit Score</label>
  <input type="number" name="credit_score"><br>

  <label>Account Age (years)</label>
  <input type="number" step="0.1" name="account_age_years"><br>

  <label>Account Balance</label>
  <input type="number" step="0.01" name="account_balance"><br>

  <label>Transaction Amount</label>
  <input type="number" step="0.01" name="transaction_amount"><br>

  <label>Number of Previous Transactions</label>
  <input type="number" name="num_prev_transactions"><br>

  <label>Transaction Frequency (Monthly)</label>
  <input type="number" step="0.1" name="transaction_freq_monthly"><br>

  <label>Distance from Home (km)</label>
  <input type="number" step="0.1" name="distance_from_home_km"><br>

  <label>Time Since Last Transaction (hrs)</label>
  <input type="number" step="0.1" name="time_since_last_txn_hrs"><br>

  <label>Failed Attempts</label>
  <input type="number" name="failed_attempts"><br>


  <!-- BOOLEAN FEATURES -->
  <label>International Transaction</label>
  <input type="checkbox" name="is_international"><br>

  <label>PIN Changed Recently</label>
  <input type="checkbox" name="pin_changed_recently"><br>


  <!-- CATEGORICAL FEATURES -->
  <label>City</label>
  <select name="city">
    <option>Delhi</option>
    <option>Guadalajara</option>
    <option>London</option>
    <option>Los Angeles</option>
    <option>Lyon</option>
    <option>Manchester</option>
    <option>Melbourne</option>
    <option>Mexico City</option>
    <option>Mumbai</option>
    <option>Munich</option>
    <option>New York</option>
    <option>Osaka</option>
    <option>Paris</option>
    <option>Rio</option>
    <option>Sydney</option>
    <option>São Paulo</option>
    <option>Tokyo</option>
    <option>Toronto</option>
    <option>Vancouver</option>
  </select><br>


  <label>Merchant Category</label>
  <select name="merchant_category">
    <option>Clothing</option>
    <option>Crypto Exchange</option>
    <option>Education</option>
    <option>Electronics</option>
    <option>Entertainment</option>
    <option>Fuel</option>
    <option>Gaming</option>
    <option>Grocery</option>
    <option>Healthcare</option>
    <option>Jewelry</option>
    <option>Online Shopping</option>
    <option>Restaurant</option>
    <option>Travel</option>
    <option>Utilities</option>
  </select><br>


  <label>Country</label>
  <select name="country">
    <option>Brazil</option>
    <option>Canada</option>
    <option>France</option>
    <option>Germany</option>
    <option>India</option>
    <option>Japan</option>
    <option>Mexico</option>
    <option>UK</option>
    <option>USA</option>
  </select><br>


  <label>Payment Method</label>
  <select name="payment_method">
    <option>Cheque</option>
    <option>Credit Card</option>
    <option>Crypto</option>
    <option>Debit Card</option>
    <option>Mobile Payment</option>
  </select><br>


  <label>Device Type</label>
  <select name="device_type">
    <option>Desktop</option>
    <option>Mobile</option>
    <option>POS Terminal</option>
    <option>Tablet</option>
  </select><br>

  <button type="submit">Submit</button>

</form>
    
    <a href='/'><button>return</button></a>"""
@app.post("/predict", response_class=HTMLResponse)
def predict(
    transaction_date: date = Form(...),
    transaction_time: str = Form(...),

    customer_age: int = Form(...),
    credit_score: int = Form(...),
    account_age_years: float = Form(...),
    account_balance: float = Form(...),
    transaction_amount: float = Form(...),
    num_prev_transactions: int = Form(...),
    transaction_freq_monthly: float = Form(...),
    distance_from_home_km: float = Form(...),
    time_since_last_txn_hrs: float = Form(...),
    failed_attempts: int = Form(...),

    is_international: bool = Form(False),
    pin_changed_recently: bool = Form(False),

    city: str = Form(...),
    merchant_category: str = Form(...),
    country: str = Form(...),
    payment_method: str = Form(...),
    device_type: str = Form(...)
):
    data = {
        "transaction_date": transaction_date,
        "transaction_time": transaction_time,
        "customer_age": customer_age,
        "credit_score": credit_score,
        "account_age_years": account_age_years,
        "account_balance": account_balance,
        "transaction_amount": transaction_amount,
        "num_prev_transactions": num_prev_transactions,
        "transaction_freq_monthly": transaction_freq_monthly,
        "distance_from_home_km": distance_from_home_km,
        "time_since_last_txn_hrs": time_since_last_txn_hrs,
        "failed_attempts": failed_attempts,
        "is_international": int(is_international),
        "pin_changed_recently": int(pin_changed_recently),
        "city": city,
        "merchant_category": merchant_category,
        "country": country,
        "payment_method": payment_method,
        "device_type": device_type,
    }
    df = pd.DataFrame([data])
    df=clean1(df)
    X=encode1(df)
    X=scaler.transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_tensor)
    probs = torch.softmax(logits, dim=1)
    probs_list=probs.detach().cpu().numpy().tolist()
    pred = torch.argmax(logits, dim=1)
    label = idx_to_label[int(pred[0])]
    return f"""<h1>Got em data!</h1>{df.to_html(index=False)}
    <a href='/'><button>Return</button></a>
    <h1>probaility list</h1>
    <pre>{probs_list}</pre>
    <h1>Prediction</h1>
    <p>It's {label}!</p>"""
ngrok.kill()
ngrok.set_auth_token('token')
public_url = ngrok.connect(8000)
print(public_url)