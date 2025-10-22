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
import plotly.express as px

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
        df = df.reset_index()
        date_col = next((c for c in df.columns if c.lower() in ['date','datetime','index','level_0']), df.columns[0])
        close_col = next((c for c in df.columns if c.lower() in ['close','adjclose','adj close','closeadj','close_adj']), None)
        if close_col is None and len(df.columns) >= 2:
            close_col = df.columns[1]
        elif close_col is None:
            return pd.DataFrame()

        df2 = pd.DataFrame()
        # convert to datetime (tz-aware) and then remove tz to make tz-naive
        df2['ds'] = pd.to_datetime(df[date_col], errors='coerce', utc=True).dt.tz_convert(None)
        df2['y'] = pd.to_numeric(df[close_col], errors='coerce')
        df2 = df2.dropna().drop_duplicates(subset='ds').sort_values('ds')
        return df2
    return pd.DataFrame()



@app.route('/dashboard', methods=['GET','POST'])
@login_required
def dashboard():
    chart = None
    pie_chart = None
    summary = []
    paragraph_summary = ""
    trade_action = None
    trade_reason = None
    stock_symbol = None
    start_date = None
    end_date = None
    filtered_chart = None
    filtered_summary = None  # <<< Initialize here

    # Initialize overview box variables
    last_actual = "-"
    forecast_next_7 = "-"
    forecast_next_30 = "-"

    if request.method == 'POST':
        stock_symbol = request.form['stock'].upper().strip()
        try:
            ticker = Ticker(stock_symbol)
            hist = ticker.history(period='2y', interval='1d')
            df = _hist_to_dataframe(hist)

            if df.empty or df['y'].isna().all():
                flash(f"❌ Could not fetch valid historical data for '{stock_symbol}'.")
                summary = ["❌ Invalid symbol or no data."]
                return render_template(
                    'dashboard.html',
                    chart=None, pie_chart=None, summary=summary, paragraph_summary="",
                    trade_action=None, trade_reason=None,
                    last_actual="-", forecast_next_7="-", forecast_next_30="-"
                )

            # Prophet prediction
            model = Prophet(daily_seasonality=True)
            model.fit(df)
            future = model.make_future_dataframe(periods=90)
            forecast = model.predict(future)

            # Actual & Predicted chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
            fig.update_layout(
                title={'text': f"{stock_symbol} - Actual & 90-day Forecast",'x':0.5,'xanchor':'center'},
                xaxis_title="Date", yaxis_title="Price",
                hovermode='x unified', template='plotly_white'
            )
            chart = plotly.io.to_html(fig, full_html=False, include_plotlyjs='cdn')

            # Overview box values
            last_actual = round(df['y'].iloc[-1], 2)
            forecast_next_7 = round(forecast['yhat'].iloc[-7:].mean(), 2) if len(forecast) >= 7 else round(forecast['yhat'].iloc[-1],2)
            forecast_next_30 = round(forecast['yhat'].iloc[-30:].mean(), 2) if len(forecast) >= 30 else round(forecast['yhat'].iloc[-1],2)

            # Prediction ranges
            max_pred = float(forecast['yhat'].max())
            min_pred = float(forecast['yhat'].min())

            overall_change = ((forecast_next_30 - last_actual)/last_actual)*100
            short_term_change = ((forecast_next_7 - last_actual)/last_actual)*100

            # Guidance points
            points = []
            points.append(f"📈 Last actual price: ${last_actual}")
            points.append(f"🔹 Short-term 7-day forecast: ${forecast_next_7}")
            points.append(f"🔹 Mid-term 30-day forecast: ${forecast_next_30}")
            points.append(f"⚡ Predicted range (min→max): ${min_pred:.2f} → ${max_pred:.2f}")

            # Advice
            if overall_change > 5:
                trade_action = "BUY"
                trade_reason = f"Forecast predicts a significant rise (~{overall_change:.2f}%)."
                points.append("💡 Advice: Consider buying or holding if your risk tolerance allows.")
            elif overall_change < -5:
                trade_action = "SELL"
                trade_reason = f"Forecast predicts a notable decline (~{overall_change:.2f}%)."
                points.append("💡 Advice: Consider reducing exposure or selling; review fundamentals/news.")
            else:
                trade_action = "HOLD"
                trade_reason = f"Forecast suggests minor change (~{overall_change:.2f}%). Monitor closely."
                points.append("💡 Advice: No strong signal — consider monitoring or small-scale positions.")

            paragraph_summary = (
                f"The forecast for {stock_symbol} suggests that over the next 30 days, the stock price may "
                f"{'rise' if overall_change>0 else 'fall' if overall_change<0 else 'remain roughly stable'}. "
                f"The 7-day and 30-day averages provide additional insight into the price trajectory."
            )

            summary = points

            # Market Segmentation Pie Chart (dummy sectors for demonstration)
            # Real Market Segmentation Pie Chart
            try:
                market = request.form.get('market', 'NASDAQ')  # Get selected market

                # Select top stocks based on market
                if market == "NASDAQ":
                    top_stocks = ['AAPL','MSFT','GOOGL','AMZN','TSLA','NVDA']
                elif market == "NYSE":
                    top_stocks = ['JPM','JNJ','XOM','PG','V','BAC']
                elif market == "NSE":
                    top_stocks = ['RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','ICICIBANK.NS']
                elif market == "BSE":
                    top_stocks = ['500325.BO','500180.BO','500209.BO','500016.BO','500112.BO']
                else:
                    top_stocks = ['AAPL','MSFT','GOOGL']

                sectors_list = []
                for sym in top_stocks:
                    try:
                        t = Ticker(sym)
                        prof = t.summary_profile.get(sym, {})
                        sector = prof.get('sector', 'Unknown')
                        sectors_list.append(sector)
                    except:
                        sectors_list.append('Unknown')

                from collections import Counter
                sector_counts = Counter(sectors_list)
                sectors = list(sector_counts.keys())
                shares = list(sector_counts.values())

                pie_fig = go.Figure(data=[go.Pie(labels=sectors, values=shares, hole=0.3)])
                pie_fig.update_layout(title=f"{market} Market Segmentation by Sector")
                pie_chart = plotly.io.to_html(pie_fig, full_html=False, include_plotlyjs='cdn')

            except Exception as e:
                pie_chart = None
                print("[ERROR] Pie chart generation failed:", e)



            # After your existing chart code in the dashboard function
            # Get date filter from form (if any)
            # Get date filter from form (if any)
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            filtered_chart = None
            filtered_summary = None

            if start_date and end_date:
                try:
                    # Filter actual and predicted data
                    df_filtered = df[(df['ds'] >= pd.to_datetime(start_date)) &
                                     (df['ds'] <= pd.to_datetime(end_date))]
                    forecast_filtered = forecast[(forecast['ds'] >= pd.to_datetime(start_date)) &
                                                 (forecast['ds'] <= pd.to_datetime(end_date))]

                    if not df_filtered.empty and not forecast_filtered.empty:
                        fig_filtered = go.Figure()
                        # Actual data
                        fig_filtered.add_trace(go.Scatter(
                            x=df_filtered['ds'], y=df_filtered['y'], mode='lines+markers', name='Actual'
                        ))
                        # Predicted data
                        fig_filtered.add_trace(go.Scatter(
                            x=forecast_filtered['ds'], y=forecast_filtered['yhat'], mode='lines+markers', name='Predicted'
                        ))
                        fig_filtered.update_layout(
                            title={'text': f"{stock_symbol} Actual vs Predicted from {start_date} to {end_date}",
                                   'x':0.5,'xanchor':'center'},
                            xaxis_title="Date", yaxis_title="Price",
                            template='plotly_white'
                        )
                        filtered_chart = plotly.io.to_html(fig_filtered, full_html=False, include_plotlyjs='cdn')

                        # Generate textual summary
                        actual_start = df_filtered['y'].iloc[0]
                        actual_end = df_filtered['y'].iloc[-1]
                        pred_start = forecast_filtered['yhat'].iloc[0]
                        pred_end = forecast_filtered['yhat'].iloc[-1]

                        change_actual = ((actual_end - actual_start) / actual_start) * 100
                        change_pred = ((pred_end - pred_start) / pred_start) * 100

                        filtered_summary = (
                            f"From {start_date} to {end_date}, the stock price actually moved from "
                            f"${actual_start:.2f} to ${actual_end:.2f} ({change_actual:+.2f}%). "
                            f"The predicted trend for this period was from ${pred_start:.2f} to ${pred_end:.2f} ({change_pred:+.2f}%). "
                            f"Compare actual vs predicted to assess accuracy and market behavior."
                        )
                    else:
                        filtered_chart = "<p>No data available for the selected range.</p>"
                        filtered_summary = None

                except Exception as e:
                    filtered_chart = f"<p>Error generating chart: {e}</p>"
                    filtered_summary = None



        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print("[ERROR]", tb)
            flash(f"❌ Error analyzing stock '{stock_symbol}': {e}")
            summary = [f"❌ Error: {e}"]
            paragraph_summary = ""
            trade_action = None
            trade_reason = None
            last_actual = forecast_next_7 = forecast_next_30 = "-"
            chart = pie_chart = None

    return render_template(
    'dashboard.html',
    chart=chart,
    pie_chart=pie_chart,
    summary=summary,
    paragraph_summary=paragraph_summary,
    trade_action=trade_action,
    trade_reason=trade_reason,
    last_actual=last_actual,
    forecast_next_7=forecast_next_7,
    forecast_next_30=forecast_next_30,
    filtered_chart=filtered_chart,
    filtered_summary=filtered_summary,
    start_date=start_date or '',
    end_date=end_date or '',
    stock_symbol=stock_symbol or ''
)






if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
