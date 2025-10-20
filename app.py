from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from prophet import Prophet
import plotly.graph_objs as go
import plotly
import pandas as pd
from yahooquery import Ticker
import traceback
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fallback_secret_key") 

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Login manager
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash("❌ Username already exists!")
            return redirect(url_for('register'))
        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        flash("✅ Registered successfully! Please login.")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash("❌ Invalid username or password")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("✅ Logged out successfully!")
    return redirect(url_for('index'))



def _hist_to_dataframe(hist):
    if isinstance(hist, dict):
        try:
            hist = next(iter(hist.values()))
        except:
            return pd.DataFrame()
    if isinstance(hist, pd.DataFrame):
        df = hist.copy()
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        else:
            df = df.reset_index()
        date_col = next((c for c in df.columns if c.lower() in ['date','datetime','index','level_0']), df.columns[0])
        close_col = next((c for c in df.columns if c.lower() in ['close','adjclose','adj close','closeadj','close_adj']), None)
        if close_col is None and len(df.columns)>=2:
            close_col = df.columns[1]
        elif close_col is None:
            return pd.DataFrame()
        df2 = pd.DataFrame()
        df2['ds'] = pd.to_datetime(df[date_col])
        df2['y'] = pd.to_numeric(df[close_col], errors='coerce')
        df2 = df2.dropna().drop_duplicates(subset='ds').sort_values('ds')
        return df2
    return pd.DataFrame()



@app.route('/dashboard', methods=['GET','POST'])
@login_required
def dashboard():
    chart = None
    summary = []
    paragraph_summary = ""
    trade_action = None
    trade_reason = None

    if request.method == 'POST':
        stock_symbol = request.form['stock'].upper().strip()
        try:
            ticker = Ticker(stock_symbol)
            hist = ticker.history(period='2y', interval='1d')
            df = _hist_to_dataframe(hist)

            if df.empty:
                flash(f"❌ Could not fetch valid historical data for '{stock_symbol}'.")
                summary = ["❌ Invalid symbol or no data."]
                return render_template('dashboard.html', chart=None, summary=summary, paragraph_summary="", trade_action=None, trade_reason=None)

            # Prophet prediction
            model = Prophet(daily_seasonality=True)
            model.fit(df)
            future = model.make_future_dataframe(periods=90)
            forecast = model.predict(future)

            # Plotly chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
            fig.update_layout(
                title={'text': f"{stock_symbol} - Actual & 90-day Forecast",'x':0.5,'xanchor':'center'},
                xaxis_title="Date", yaxis_title="Price",
                hovermode='x unified', template='plotly_white',
                margin=dict(l=60,r=60,t=100,b=60)
            )

            chart = plotly.io.to_html(
                fig, full_html=False, include_plotlyjs='cdn',
                config={
                    'displaylogo':False,
                    'responsive':True,
                    'scrollZoom':True,
                    'displayModeBar':True,
                    'modeBarButtonsToRemove':[],
                    'toImageButtonOptions':{'format':'png','filename':f'{stock_symbol}_forecast_chart'},
                    'modeBarPosition':'topRight'
                }
            )

            
            last_actual = df['y'].iloc[-1]
            forecast_next_7 = forecast['yhat'].iloc[-7:].mean() if len(forecast)>=7 else forecast['yhat'].iloc[-1]
            forecast_next_30 = forecast['yhat'].iloc[-30:].mean() if len(forecast)>=30 else forecast['yhat'].iloc[-1]
            forecast_next_90 = forecast['yhat'].iloc[-90:].mean() if len(forecast)>=90 else forecast['yhat'].iloc[-1]

            max_pred = float(forecast['yhat'].max())
            min_pred = float(forecast['yhat'].min())

            overall_change = ((forecast_next_90-last_actual)/last_actual)*100
            short_term_change = ((forecast_next_7-last_actual)/last_actual)*100

            points = []

            # Long-term trend
            if overall_change > 5:
                points.append(f"📈 Expected rise: ~{overall_change:.2f}% over the next 90 days.")
            elif overall_change < -5:
                points.append(f"📉 Expected drop: ~{overall_change:.2f}% over the next 90 days.")
            else:
                points.append(f"➡️ Expected minor change: ~{overall_change:.2f}% over the next 90 days.")

            # Short-term trend
            points.append(f"🔹 Short-term (7 days) trend: {'up' if short_term_change>0 else 'down' if short_term_change<0 else 'flat'} (~{short_term_change:.2f}%).")

            # Predicted range
            points.append(f"⚡ Predicted range (min→max): ${min_pred:.2f} → ${max_pred:.2f}.")

            # Advice based on trend
            if overall_change > 5:
                points.append("💡 Advice: Consider buying or holding if your risk tolerance allows.")
            elif overall_change < -5:
                points.append("💡 Advice: Consider reducing exposure or selling; review fundamentals/news.")
            else:
                points.append("💡 Advice: No strong signal — consider monitoring or small-scale positions.")

            paragraph_summary = (
                f"The forecast for {stock_symbol} suggests that over the next 90 days, the stock price may "
                f"{'rise' if overall_change>0 else 'fall' if overall_change<0 else 'remain roughly stable'}. "
                f"The short-term 7-day trend indicates potential movements, while the 30-day and 90-day averages "
                f"provide additional insight into the price trajectory. Always combine these predictions with company fundamentals, "
                "news, and broader market conditions. Never invest money you cannot afford to lose."
            )

            # Trade recommendation
            if overall_change > 5:
                trade_action = "BUY"
                trade_reason = f"Forecast predicts a significant rise (~{overall_change:.2f}%)."
            elif overall_change < -5:
                trade_action = "SELL"
                trade_reason = f"Forecast predicts a notable decline (~{overall_change:.2f}%)."
            else:
                trade_action = "HOLD"
                trade_reason = f"Forecast suggests minor change (~{overall_change:.2f}%). Monitor closely."

            summary = points

        except Exception as e:
            tb = traceback.format_exc()
            print("[ERROR]", tb)
            flash(f"❌ Error analyzing stock '{stock_symbol}': {e}")
            summary = [f"❌ Error: {e}"]
            paragraph_summary = ""
            trade_action = None
            trade_reason = None

    return render_template(
        'dashboard.html',
        chart=chart,
        summary=summary,
        paragraph_summary=paragraph_summary,
        trade_action=trade_action,
        trade_reason=trade_reason
    )




if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
