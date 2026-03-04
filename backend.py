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

def bs_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes Vanna und Charm.
    S = spot price, K = strike, T = time in years,
    r = risk-free rate, sigma = implied vol
    Returns: (vanna, charm)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        pdf_d1 = norm.pdf(d1)

        # Vanna = dDelta/dSigma = dVega/dS
        # = -pdf(d1) * d2 / sigma
        vanna = -pdf_d1 * d2 / sigma

        # Charm = dDelta/dTime (per day, negative = delta decays)
        # Call charm:  -pdf(d1) * (2*r*T - d2*sigma*sqrt(T)) / (2*T*sigma*sqrt(T))
        # Put charm = call charm (same formula, put delta is call delta - 1)
        charm_raw = -pdf_d1 * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
        # Convert to per-day
        charm = charm_raw / 365

        return float(vanna), float(charm)
    except Exception:
        return 0.0, 0.0

@app.route('/api/options/<ticker>')
def get_options(ticker):
    try:
        num_strikes = int(request.args.get('strikes', 50))
        selected_dte = int(request.args.get('dte', 0))  # Welches DTE soll angezeigt werden

        print(f"\n📡 Lade Daten für {ticker}, DTE={selected_dte}...")

        stock = yf.Ticker(ticker)
        current_price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')

        if not current_price:
            hist = stock.history(period='1d')
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])

        if not current_price:
            return jsonify({'error': 'Kein aktueller Preis verfügbar'}), 404

        print(f"✅ Aktueller Preis: ${current_price}")

        all_expirations = stock.options
        if not all_expirations:
            return jsonify({'error': 'Keine Options-Daten verfügbar'}), 404

        # Timezone-korrektes heutiges Datum (US Eastern)
        today = get_eastern_today()
        print(f"📅 Heute (Eastern): {today}")

        # Finde verfügbare DTEs (0-7)
        available_dtes = {}
        for exp_date in all_expirations[:20]:
            exp_day = datetime.strptime(exp_date, '%Y-%m-%d').date()
            dte = (exp_day - today).days
            if 0 <= dte <= 7 and dte not in available_dtes:
                available_dtes[dte] = exp_date
                print(f"  ✅ {exp_date} = {dte}DTE")

        if not available_dtes:
            return jsonify({
                'error': f'Keine 0-7 DTE Expirations. Nächste: {list(all_expirations[:5])}'
            }), 404

        # Wenn gewünschtes DTE nicht verfügbar, nimm nächstes
        if selected_dte not in available_dtes:
            selected_dte = min(available_dtes.keys())
            print(f"⚠️ DTE {selected_dte} nicht gefunden, nutze: {selected_dte}")

        exp_date = available_dtes[selected_dte]
        opt = stock.option_chain(exp_date)
        calls = opt.calls
        puts = opt.puts

        print(f"📊 Lade {exp_date} ({selected_dte}DTE): {len(calls)} Calls, {len(puts)} Puts")
        print(f"   Verfügbare Greeks: {[c for c in calls.columns if c in ['delta','gamma','vega','theta','vanna','charm']]}")

        # Strikes auswählen
        available_strikes = sorted(set(
            list(calls['strike'].unique()) +
            list(puts['strike'].unique())
        ))

        closest_strike = min(available_strikes, key=lambda x: abs(x - current_price))
        closest_idx = available_strikes.index(closest_strike)
        half = num_strikes // 2
        start_idx = max(0, closest_idx - half)
        end_idx = min(len(available_strikes), closest_idx + half + 1)
        strikes = [int(round(s)) for s in available_strikes[start_idx:end_idx]]

        print(f"✅ {len(strikes)} Strikes: {min(strikes)} - {max(strikes)}")

        # Berechne GEX, DEX, Charm für jeden Strike
        exposure_data = []
        for strike in strikes:
            call_data = calls[calls['strike'] == strike]
            put_data = puts[puts['strike'] == strike]

            call_oi = int(call_data['openInterest'].sum()) if not call_data.empty else 0
            put_oi = int(put_data['openInterest'].sum()) if not put_data.empty else 0

            # GEX
            if 'gamma' in calls.columns:
                call_gamma = call_data['gamma'].sum() if not call_data.empty else 0
                put_gamma = put_data['gamma'].sum() if not put_data.empty else 0
                gex = round(((call_oi * call_gamma) - (put_oi * put_gamma)) * 100 / 1_000_000, 2)
            else:
                gex = round((call_oi - put_oi) / 10000, 2)

            # DEX
            if 'delta' in calls.columns:
                call_delta = call_data['delta'].sum() if not call_data.empty else 0
                put_delta = put_data['delta'].sum() if not put_data.empty else 0
                dex = round(((call_oi * call_delta) + (put_oi * put_delta)) * 100 / 1_000_000, 2)
            else:
                dex = round((call_oi - put_oi) / 10000, 2)

            # Black-Scholes Vanna + Charm pro Strike
            # Nutze impliedVolatility aus yfinance, berechne BS greeks selbst
            T = max(selected_dte, 0.5) / 365  # Zeit in Jahren
            r = 0.05  # Risk-free rate (~aktueller Fed Rate)

            call_iv = float(call_data['impliedVolatility'].mean()) if not call_data.empty and 'impliedVolatility' in call_data.columns else 0.2
            put_iv  = float(put_data['impliedVolatility'].mean())  if not put_data.empty  and 'impliedVolatility' in put_data.columns  else 0.2
            call_iv = call_iv if call_iv > 0 else 0.2
            put_iv  = put_iv  if put_iv  > 0 else 0.2

            c_vanna, c_charm = bs_greeks(current_price, strike, T, r, call_iv, 'call')
            p_vanna, p_charm = bs_greeks(current_price, strike, T, r, put_iv,  'put')

            # VEX: (Call_OI * call_vanna - Put_OI * put_vanna) * 100 contracts / 1M scaling
            vex   = round(((call_oi * c_vanna) - (put_oi * p_vanna)) * 100 / 1_000_000, 2)
            # Charm Exposure: same structure as GEX but with charm greek
            charm = round(((call_oi * c_charm) - (put_oi * p_charm)) * 100 / 1_000_000, 2)

            exposure_data.append({
                'strike': strike,
                'gex': gex,
                'dex': dex,
                'vex': vex,
                'charm': charm,
                'call_oi': call_oi,
                'put_oi': put_oi,
            })

        response = {
            'ticker': ticker,
            'currentPrice': current_price,
            'data': exposure_data,
            'selectedDte': selected_dte,
            'expDate': exp_date,
            'availableDtes': sorted(available_dtes.keys()),
            'timestamp': datetime.now(pytz.timezone('US/Eastern')).isoformat(),
        }

        print(f"✅ Fertig: {len(exposure_data)} Strikes\n")
        return jsonify(response)

    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    eastern = pytz.timezone('US/Eastern')
    now_eastern = datetime.now(eastern)
    return jsonify({
        'status': 'ok',
        'server_time_utc': datetime.utcnow().isoformat(),
        'server_time_eastern': now_eastern.isoformat(),
        'today_eastern': get_eastern_today().isoformat(),
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
