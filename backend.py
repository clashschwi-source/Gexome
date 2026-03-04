from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime
import pytz

app = Flask(__name__)
CORS(app)

def get_eastern_today():
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern).date()

def bs_greeks(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        pdf_d1 = norm.pdf(d1)
        vanna = -pdf_d1 * d2 / sigma
        charm = -pdf_d1 * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T)) / 365
        return float(vanna), float(charm)
    except:
        return 0.0, 0.0

def get_safe_iv(data, col, fallback):
    if data is None or data.empty or col not in data.columns:
        return fallback
    v = data[col].mean()
    if v is None or (isinstance(v, float) and (np.isnan(v) or v <= 0)):
        return fallback
    return float(v)

@app.route('/api/options/<ticker>')
def get_options(ticker):
    try:
        num_strikes = int(request.args.get('strikes', 50))
        selected_dte = int(request.args.get('dte', 0))

        stock = yf.Ticker(ticker)
        current_price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')
        if not current_price:
            hist = stock.history(period='1d')
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
        if not current_price:
            return jsonify({'error': 'Kein aktueller Preis verfügbar'}), 404

        all_expirations = stock.options
        if not all_expirations:
            return jsonify({'error': 'Keine Options-Daten verfügbar'}), 404

        today = get_eastern_today()
        available_dtes = {}
        for exp_date in all_expirations[:20]:
            exp_day = datetime.strptime(exp_date, '%Y-%m-%d').date()
            dte = (exp_day - today).days
            if 0 <= dte <= 7 and dte not in available_dtes:
                available_dtes[dte] = exp_date

        if not available_dtes:
            return jsonify({'error': f'Keine 0-7 DTE. Nächste: {list(all_expirations[:5])}'}), 404

        if selected_dte not in available_dtes:
            selected_dte = min(available_dtes.keys())

        exp_date = available_dtes[selected_dte]
        opt = stock.option_chain(exp_date)
        calls = opt.calls
        puts = opt.puts

        # ATM IV als globaler Fallback
        atm_iv = 0.2
        if 'impliedVolatility' in calls.columns:
            med = calls['impliedVolatility'].median()
            if med and not np.isnan(med) and med > 0:
                atm_iv = float(med)

        available_strikes = sorted(set(list(calls['strike'].unique()) + list(puts['strike'].unique())))
        closest_idx = min(range(len(available_strikes)), key=lambda i: abs(available_strikes[i] - current_price))
        half = num_strikes // 2
        strikes = [int(round(s)) for s in available_strikes[max(0, closest_idx-half):closest_idx+half+1]]

        T = max(selected_dte, 0.5) / 365
        r = 0.05
        exposure_data = []

        for strike in strikes:
            call_data = calls[calls['strike'] == strike]
            put_data  = puts[puts['strike'] == strike]

            call_oi = int(call_data['openInterest'].sum()) if not call_data.empty else 0
            put_oi  = int(put_data['openInterest'].sum())  if not put_data.empty  else 0

            # GEX (unverändert von yfinance)
            if 'gamma' in calls.columns:
                cg = call_data['gamma'].sum() if not call_data.empty else 0
                pg = put_data['gamma'].sum()  if not put_data.empty  else 0
                gex = round(((call_oi * cg) - (put_oi * pg)) * 100 / 1_000_000, 2)
            else:
                gex = round((call_oi - put_oi) / 10000, 2)

            # DEX (unverändert von yfinance)
            if 'delta' in calls.columns:
                cd  = call_data['delta'].sum() if not call_data.empty else 0
                pd_ = put_data['delta'].sum()  if not put_data.empty  else 0
                dex = round(((call_oi * cd) + (put_oi * pd_)) * 100 / 1_000_000, 2)
            else:
                dex = round((call_oi - put_oi) / 10000, 2)

            # IV per Strike (mit ATM Fallback)
            call_iv = get_safe_iv(call_data, 'impliedVolatility', atm_iv)
            put_iv  = get_safe_iv(put_data,  'impliedVolatility', atm_iv)
            mid_iv  = round((call_iv + put_iv) / 2, 4)

            # Black-Scholes: Vanna + Charm
            c_vanna, c_charm = bs_greeks(current_price, strike, T, r, call_iv)
            p_vanna, p_charm = bs_greeks(current_price, strike, T, r, put_iv)

            vex   = round(((call_oi * c_vanna) - (put_oi * p_vanna)) * 100 / 1_000_000, 2)
            charm = round(((call_oi * c_charm) - (put_oi * p_charm)) * 100 / 1_000_000, 2)

            exposure_data.append({
                'strike': strike,
                'gex': gex, 'dex': dex, 'vex': vex, 'charm': charm,
                'call_oi': call_oi, 'put_oi': put_oi,
                'call_iv': round(call_iv, 4),
                'put_iv':  round(put_iv,  4),
                'mid_iv':  mid_iv,
            })

        return jsonify({
            'ticker': ticker,
            'currentPrice': current_price,
            'data': exposure_data,
            'selectedDte': selected_dte,
            'expDate': exp_date,
            'availableDtes': sorted(available_dtes.keys()),
            'timestamp': datetime.now(pytz.timezone('US/Eastern')).isoformat(),
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/smile/<ticker>')
def get_smile(ticker):
    """Vol Smile für 0-5 DTE"""
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')
        if not current_price:
            hist = stock.history(period='1d')
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])

        all_expirations = stock.options
        today = get_eastern_today()

        available_dtes = {}
        for exp_date in all_expirations[:20]:
            exp_day = datetime.strptime(exp_date, '%Y-%m-%d').date()
            dte = (exp_day - today).days
            if 0 <= dte <= 5 and dte not in available_dtes:
                available_dtes[dte] = exp_date

        series = []
        for dte in sorted(available_dtes.keys()):
            exp_date = available_dtes[dte]
            try:
                opt = stock.option_chain(exp_date)
                calls = opt.calls
                puts  = opt.puts

                if 'impliedVolatility' not in calls.columns:
                    continue

                all_strikes = sorted(set(list(calls['strike'].unique()) + list(puts['strike'].unique())))
                # ±15% um Spot
                atm_strikes = [s for s in all_strikes if 0.85 * current_price <= s <= 1.15 * current_price]

                points = []
                for strike in atm_strikes:
                    cd  = calls[calls['strike'] == strike]
                    pd_ = puts[puts['strike'] == strike]

                    c_iv = get_safe_iv(cd,  'impliedVolatility', None)
                    p_iv = get_safe_iv(pd_, 'impliedVolatility', None)

                    # OTM convention: puts unter Spot, calls über Spot
                    if strike >= current_price and c_iv and c_iv > 0:
                        iv = c_iv
                    elif strike < current_price and p_iv and p_iv > 0:
                        iv = p_iv
                    elif c_iv and c_iv > 0:
                        iv = c_iv
                    elif p_iv and p_iv > 0:
                        iv = p_iv
                    else:
                        continue

                    # Filter unrealistische IV Werte (yfinance Garbage bei tief OTM)
                    iv_pct = float(iv) * 100
                    if iv_pct < 3 or iv_pct > 200:
                        continue
                    # Mindest-Volumen/OI Filter
                    cd_oi  = int(cd['openInterest'].sum())  if not cd.empty  and 'openInterest'  in cd.columns  else 0
                    pd_oi  = int(pd_['openInterest'].sum()) if not pd_.empty and 'openInterest'  in pd_.columns else 0
                    cd_vol = int(cd['volume'].sum())        if not cd.empty  and 'volume'        in cd.columns  else 0
                    pd_vol = int(pd_['volume'].sum())       if not pd_.empty and 'volume'        in pd_.columns else 0
                    if cd_oi + pd_oi < 5 and cd_vol + pd_vol < 2:
                        continue

                    moneyness = round(float(np.log(strike / current_price)), 4)
                    points.append({
                        'strike': int(strike),
                        'moneyness': moneyness,
                        'iv': round(iv_pct, 2),
                    })

                if points:
                    series.append({
                        'dte': dte,
                        'expDate': exp_date,
                        'label': f'{dte}DTE',
                        'points': sorted(points, key=lambda x: x['strike']),
                    })
            except Exception as e:
                print(f"Smile Fehler {exp_date}: {e}")
                continue

        return jsonify({
            'ticker': ticker,
            'currentPrice': current_price,
            'series': series,
            'timestamp': datetime.now(pytz.timezone('US/Eastern')).isoformat(),
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    eastern = pytz.timezone('US/Eastern')
    now_e = datetime.now(eastern)
    return jsonify({
        'status': 'ok',
        'server_time_eastern': now_e.isoformat(),
        'today_eastern': get_eastern_today().isoformat(),
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
