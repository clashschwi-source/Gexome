from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

@app.route('/api/options/<ticker>')
def get_options(ticker):
    try:
        # Parameter aus Request
        num_strikes = int(request.args.get('strikes', 50))
        strike_step = int(request.args.get('strike_step', 1))
        exposure_type = request.args.get('exposure', 'gamma')  # gamma, delta, oder vanna
        
        print(f"\n📡 Lade Daten für {ticker} (Strikes: {num_strikes}, Step: {strike_step}, Exposure: {exposure_type})...")
        
        # Hole Ticker Daten
        stock = yf.Ticker(ticker)
        current_price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')
        
        if not current_price:
            return jsonify({'error': 'Kein aktueller Preis verfügbar'}), 404
        
        print(f"✅ Aktueller Preis: ${current_price}")
        
        # Hole nur 0DTE, 1DTE, 2DTE Expirations
        all_expirations = stock.options
        
        if not all_expirations:
            return jsonify({'error': 'Keine Options-Daten verfügbar'}), 404
        
        # Sammle alle Expirations mit DTE
        today = datetime.now()
        exp_with_dte = []
        
        for exp_date in all_expirations[:15]:
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
            dte = (exp_datetime - today).days
            
            if dte >= 0:  # Nur zukünftige Expirations
                exp_with_dte.append({
                    'date': exp_date,
                    'dte': dte
                })
        
        # Sortiere nach DTE (aufsteigend: 0, 1, 2, ...)
        exp_with_dte.sort(key=lambda x: x['dte'])
        
        # Nimm die ersten 3 (0DTE, 1DTE, 2DTE)
        expirations = []
        for item in exp_with_dte[:3]:
            expirations.append(item['date'])
            print(f"  ✅ Gefunden: {item['date']} ({item['dte']}DTE)")
        
        if not expirations:
            return jsonify({'error': 'Keine Options verfügbar'}), 404
        
        # Generiere Strike Range mit variablem Step
        strike_range = num_strikes * strike_step / 2
        strikes = np.arange(
            current_price - strike_range,
            current_price + strike_range + strike_step,
            strike_step
        )
        # Runde auf ganze Zahlen
        strikes = [int(round(s)) for s in strikes]
        
        print(f"✅ Strikes generiert: {len(strikes)} Strikes von {min(strikes)} bis {max(strikes)}")
        
        # Sammle Exposure Daten - OPTIMIERT: Nur 1x pro Expiration laden!
        exposure_data = []
        dates = []
        labels = []
        options_cache = {}  # Cache für Options Chains
        
        for idx, exp_date in enumerate(expirations):
            try:
                # Lade Options Chain nur 1x pro Expiration
                opt = stock.option_chain(exp_date)
                options_cache[exp_date] = {
                    'calls': opt.calls,
                    'puts': opt.puts
                }
                
                # Berechne DTE (Days to Expiration)
                exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                dte = (exp_datetime - datetime.now()).days
                
                dates.append(exp_date)
                labels.append(f"{dte}DTE")
                
                print(f"  📅 {exp_date} ({dte}DTE) - Calls: {len(opt.calls)}, Puts: {len(opt.puts)}")
                
            except Exception as e:
                print(f"  ⚠️ Fehler bei {exp_date}: {e}")
                continue
        
        if not dates:
            return jsonify({'error': 'Keine gültigen 0-2 DTE Expirations gefunden'}), 404
        
        # Erstelle Exposure Matrix - verwende gecachte Daten
        for strike in strikes:
            row = {'strike': strike}
            
            for exp_date in dates:
                try:
                    calls = options_cache[exp_date]['calls']
                    puts = options_cache[exp_date]['puts']
                    
                    # Finde nächsten verfügbaren Strike
                    call_data = calls[calls['strike'] == strike]
                    put_data = puts[puts['strike'] == strike]
                    
                    if call_data.empty and put_data.empty:
                        row[exp_date] = 0
                        continue
                    
                    # Hole Open Interest
                    call_oi = call_data['openInterest'].sum() if not call_data.empty else 0
                    put_oi = put_data['openInterest'].sum() if not put_data.empty else 0
                    
                    # Berechne basierend auf Exposure Type
                    if exposure_type == 'delta':
                        # DELTA EXPOSURE (DEX)
                        if 'delta' in calls.columns and 'delta' in puts.columns:
                            call_delta = call_data['delta'].sum() if not call_data.empty else 0
                            put_delta = put_data['delta'].sum() if not put_data.empty else 0
                            dex = ((call_oi * call_delta) + (put_oi * put_delta)) * 100 / 1_000_000
                            row[exp_date] = round(dex, 2)
                        else:
                            # Fallback ohne Delta
                            row[exp_date] = round((call_oi - put_oi) / 10000, 2)
                    
                    elif exposure_type == 'vanna':
                        # VANNA EXPOSURE (VEX)
                        if 'vanna' in calls.columns and 'vanna' in puts.columns:
                            call_vanna = call_data['vanna'].sum() if not call_data.empty else 0
                            put_vanna = put_data['vanna'].sum() if not put_data.empty else 0
                            vex = ((call_oi * call_vanna) - (put_oi * put_vanna)) * 100 / 1_000_000
                            row[exp_date] = round(vex, 2)
                        else:
                            # Fallback: Vanna oft nicht verfügbar in yfinance
                            row[exp_date] = 0
                    
                    else:  # gamma (default)
                        # GAMMA EXPOSURE (GEX)
                        if 'gamma' in calls.columns and 'gamma' in puts.columns:
                            call_gamma = call_data['gamma'].sum() if not call_data.empty else 0
                            put_gamma = put_data['gamma'].sum() if not put_data.empty else 0
                            gex = ((call_oi * call_gamma) - (put_oi * put_gamma)) * 100 / 1_000_000
                            row[exp_date] = round(gex, 2)
                        else:
                            # Fallback ohne Gamma
                            row[exp_date] = round((call_oi - put_oi) / 10000, 2)
                    
                except Exception as e:
                    print(f"    ⚠️ Fehler bei Strike {strike}, {exp_date}: {e}")
                    row[exp_date] = 0
            
            exposure_data.append(row)
        
        response = {
            'ticker': ticker,
            'currentPrice': current_price,
            'data': exposure_data,
            'dates': dates,
            'labels': labels,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Daten erfolgreich geladen: {len(exposure_data)} Strikes × {len(dates)} Expirations\n")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'Backend läuft!'})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Gexome Backend gestartet")
    print("="*50)
    print("📡 API läuft auf: http://localhost:5000")
    print("🔍 Health Check: http://localhost:5000/health")
    print("📊 Options API: http://localhost:5000/api/options/QQQ")
    print("="*50 + "\n")
    
    app.run(debug=True, port=5000)
    