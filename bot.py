import os, json, logging, threading, time, requests
from datetime import datetime
import pandas as pd
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

TOKEN   = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ZONES_FILE    = "zones.json"
NOTIFIED_FILE = "notified.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
bot = telebot.TeleBot(TOKEN)
user_state = {}

# ─── ZONES ───────────────────────────────────
def load_zones():
    try:
        if os.path.exists(ZONES_FILE):
            with open(ZONES_FILE) as f: return json.load(f)
    except: pass
    return []

def save_zones(z):
    with open(ZONES_FILE,"w") as f: json.dump(z,f,indent=2)

def load_notified():
    try:
        if os.path.exists(NOTIFIED_FILE):
            with open(NOTIFIED_FILE) as f: return json.load(f)
    except: pass
    return {}

def save_notified(n):
    with open(NOTIFIED_FILE,"w") as f: json.dump(n,f,indent=2)

# ─── TIMEFRAMES ──────────────────────────────
SUB_TFS = {
    "H4":["H1","M30","M15"],
    "H1":["M30","M15","M5"],
    "M30":["M15","M5","M1"],
}

BINANCE_PAIRS = {"BTCUSD","ETHUSD","BNBUSD","SOLUSD","XRPUSD"}

BINANCE_TF = {
    "H4":"4h","H1":"1h","M30":"30m",
    "M15":"15m","M5":"5m","M1":"1m"
}
FOREX_TF = {
    "H4":240,"H1":60,"M30":30,
    "M15":15,"M5":5,"M1":1
}

# ─── BINANCE DATA ────────────────────────────
def binance_symbol(pair):
    return pair.upper().replace("USD","USDT")

def get_binance_price(pair):
    try:
        sym = binance_symbol(pair)
        r   = requests.get(
            f"https://api.binance.com/api/v3/ticker/price?symbol={sym}",
            timeout=10
        )
        if r.status_code == 200:
            return float(r.json()["price"])
    except Exception as e:
        logging.warning(f"Binance price {pair}: {e}")
    return None

def get_binance_klines(pair, tf, limit=100):
    try:
        sym      = binance_symbol(pair)
        interval = BINANCE_TF[tf]
        r = requests.get(
            f"https://api.binance.com/api/v3/klines?symbol={sym}&interval={interval}&limit={limit}",
            timeout=10
        )
        if r.status_code != 200: return None
        data = r.json()
        df   = pd.DataFrame(data, columns=[
            "time","open","high","low","close","volume",
            "close_time","qav","trades","tbav","tqav","ignore"
        ])
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        df.columns = [c.capitalize() for c in df.columns]
        df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
        return df[["Open","High","Low","Close","Volume"]]
    except Exception as e:
        logging.warning(f"Binance klines {pair} {tf}: {e}")
        return None

# ─── FOREX DATA (twelve data) ────────────────
def get_forex_price(pair):
    try:
        sym = pair[:3] + "/" + pair[3:]
        r   = requests.get(
            f"https://api.twelvedata.com/price?symbol={sym}&apikey=demo",
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            if "price" in data:
                return float(data["price"])
    except Exception as e:
        logging.warning(f"Forex price {pair}: {e}")
    return None

def get_forex_klines(pair, tf, limit=100):
    try:
        sym      = pair[:3] + "/" + pair[3:]
        interval = f"{FOREX_TF[tf]}min" if tf not in ["H1","H4"] else ("1h" if tf=="H1" else "4h")
        r = requests.get(
            f"https://api.twelvedata.com/time_series?symbol={sym}&interval={interval}&outputsize={limit}&apikey=demo",
            timeout=10
        )
        if r.status_code != 200: return None
        data = r.json()
        if "values" not in data: return None
        rows = data["values"]
        df = pd.DataFrame(rows)
        df = df.rename(columns={
            "open":"Open","high":"High","low":"Low",
            "close":"Close","volume":"Volume"
        })
        for col in ["Open","High","Low","Close"]:
            df[col] = df[col].astype(float)
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(float)
        else:
            df["Volume"] = 0.0
        return df[["Open","High","Low","Close","Volume"]].iloc[::-1].reset_index(drop=True)
    except Exception as e:
        logging.warning(f"Forex klines {pair} {tf}: {e}")
        return None

# ─── PRIX ET DONNÉES UNIFIÉS ─────────────────
def get_price(pair):
    if pair.upper() in BINANCE_PAIRS:
        return get_binance_price(pair)
    return get_forex_price(pair)

def get_data(pair, tf):
    if pair.upper() in BINANCE_PAIRS:
        df = get_binance_klines(pair, tf)
    else:
        df = get_forex_klines(pair, tf)
    if df is None or len(df) < 20:
        return None
    return df

# ─── INDICATEURS ─────────────────────────────
def calc_rsi(closes, period=14):
    delta = closes.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return float((100-(100/(1+gain/(loss+1e-10)))).iloc[-1])

def rsi_signal(v):
    if v < 30:  return 1
    if v <= 50: return -1
    if v <= 70: return 1
    return -1

def ma10_signal(o, c, ma):
    bt, bb = max(o,c), min(o,c)
    if bb > ma: return 1
    if bt < ma: return -1
    return 0

def ich_signal(df):
    try:
        hi,lo,cl = df["High"],df["Low"],df["Close"]
        t  = (hi.rolling(9).max()+lo.rolling(9).min())/2
        k  = (hi.rolling(26).max()+lo.rolling(26).min())/2
        sa = ((t+k)/2).shift(26)
        sb = ((hi.rolling(52).max()+lo.rolling(52).min())/2).shift(26)
        sa_v = sa.dropna()
        sb_v = sb.dropna()
        if sa_v.empty or sb_v.empty: return 0
        top = max(float(sa_v.iloc[-1]),float(sb_v.iloc[-1]))
        bot = min(float(sa_v.iloc[-1]),float(sb_v.iloc[-1]))
        c   = float(cl.iloc[-1])
        if c > top: return 1
        if c < bot: return -1
        return 0
    except: return 0

def get_indicators(pair, tf):
    df = get_data(pair, tf)
    if df is None: return None
    try:
        rsi_v = calc_rsi(df["Close"])
        rsi   = rsi_signal(rsi_v)
        ma_v  = float(df["Close"].rolling(10).mean().iloc[-1])
        ma    = ma10_signal(float(df["Open"].iloc[-1]),float(df["Close"].iloc[-1]),ma_v)
        ich   = ich_signal(df)
        logging.info(f"{pair} {tf} RSI:{rsi_v:.1f}({rsi}) MA:{ma} ICH:{ich}")
        return {"rsi":rsi,"ma":ma,"ich":ich,"rsi_val":rsi_v}
    except Exception as e:
        logging.warning(f"indicators {pair} {tf}: {e}")
        return None

# ─── DÉTECTIONS ──────────────────────────────
def detect_inside_bar(pair, tf):
    df = get_data(pair, tf)
    if df is None or len(df) < 2: return False
    return bool(float(df["High"].iloc[-1]) < float(df["High"].iloc[-2]) and
                float(df["Low"].iloc[-1])  > float(df["Low"].iloc[-2]))

def detect_sl_hunt(pair, tf):
    df = get_data(pair, tf)
    if df is None or len(df) < 11: return False
    ph   = float(df["High"].iloc[-11:-1].max())
    pl   = float(df["Low"].iloc[-11:-1].min())
    up   = float(df["High"].iloc[-1]) > ph and float(df["Close"].iloc[-1]) < ph
    down = float(df["Low"].iloc[-1])  < pl and float(df["Close"].iloc[-1]) > pl
    return bool(up or down)

def detect_divergence(pair, tf):
    df = get_data(pair, tf)
    if df is None or len(df) < 6: return False
    pu = float(df["Close"].iloc[-1]) > float(df["Close"].iloc[-6])
    vu = float(df["Volume"].iloc[-1]) > float(df["Volume"].iloc[-6])
    return bool((pu and not vu) or (not pu and not vu))

# ─── ANALYSE ZONE ────────────────────────────
def analyze_zone(zone):
    pair      = zone["pair"]
    direction = zone["direction"]
    ob_tf     = zone["timeframe"]
    high      = float(zone["high"])
    low       = float(zone["low"])

    price = get_price(pair)
    if price is None:
        logging.warning(f"Prix indisponible {pair}")
        return None

    logging.info(f"Prix {pair}: {price} Zone: {low}-{high}")

    if not (low <= price <= high):
        logging.info(f"{pair} hors zone")
        return None

    logging.info(f"✅ {pair} DANS zone! Analyse...")
    sub_tfs   = SUB_TFS.get(ob_tf, ["M30","M15","M5"])
    dv        = 1 if direction=="BUY" else -1
    results   = {}
    confirmed = 0

    for tf in sub_tfs:
        ind = get_indicators(pair, tf)
        if ind is None:
            results[tf] = None
            continue
        score = 0
        if ind["rsi"] == dv: score += 1
        if ind["ma"] != 0 and ind["ma"] == dv: score += 1
        if ind["ich"] != 0 and ind["ich"] == dv: score += 1
        ok = score >= 2
        if ok: confirmed += 1
        results[tf] = {**ind,"confirmed":ok,"score":score}
        logging.info(f"  {tf}: score={score} ok={ok}")

    logging.info(f"Confirmés: {confirmed}/3")
    if confirmed < 2: return None

    return {
        "pair":pair,"direction":direction,"ob_tf":ob_tf,
        "price":price,"results":results,
        "inside_bar":detect_inside_bar(pair,ob_tf),
        "sl_hunt":detect_sl_hunt(pair,ob_tf),
        "divergence":detect_divergence(pair,ob_tf),
    }

# ─── MESSAGE ─────────────────────────────────
def format_signal(sig):
    emoji = "🟢" if sig["direction"]=="BUY" else "🔴"
    dv    = 1 if sig["direction"]=="BUY" else -1
    lines = [
        f"{emoji} SIGNAL {sig['direction']} — {sig['pair']} — OB {sig['ob_tf']}",
        "━━━━━━━━━━━━━━━━━━",
    ]
    for tf, r in sig["results"].items():
        if r is None:
            lines.append(f"  {tf}: ❓ Données indisponibles")
            continue
        rsi_e = "✅" if r["rsi"]==dv else "❌"
        ma_e  = "✅" if r["ma"]==dv  else ("⚪" if r["ma"]==0  else "❌")
        ich_e = "✅" if r["ich"]==dv else ("⚪" if r["ich"]==0 else "❌")
        ok_e  = "✅ VALIDÉ" if r["confirmed"] else "❌"
        lines.append(f"  {tf}: RSI{rsi_e}({r.get('rsi_val',0):.0f}) MA{ma_e} Ichi{ich_e} → {ok_e}")
    lines += [
        "━━━━━━━━━━━━━━━━━━",
        f"🔍 Chasse SL  : {'✅ Détectée'   if sig['sl_hunt']    else '❌ Absente'}",
        f"📦 Inside Bar : {'✅ Présente'   if sig['inside_bar'] else '❌ Absente'}",
        f"📊 Divergence : {'⚠️ Divergence' if sig['divergence'] else '✅ Convergence'}",
        "━━━━━━━━━━━━━━━━━━",
        f"💰 Prix : {sig['price']:.5f}",
        "⚠️ Place ton SL et TP manuellement",
    ]
    return "\n".join(lines)

# ─── SURVEILLANCE ────────────────────────────
def surveillance_loop():
    time.sleep(15)
    logging.info("✅ Surveillance démarrée")
    while True:
        try:
            zones    = load_zones()
            notified = load_notified()
            logging.info(f"🔍 {len(zones)} zone(s)...")
            for zone in zones:
                zid = zone.get("id","")
                try:
                    sig = analyze_zone(zone)
                    if sig:
                        if notified.get(zid): continue
                        msg = format_signal(sig)
                        kb  = InlineKeyboardMarkup()
                        kb.row(
                            InlineKeyboardButton("✅ J'entre",  callback_data=f"enter_{zid}"),
                            InlineKeyboardButton("❌ Je passe", callback_data=f"skip_{zid}")
                        )
                        bot.send_message(CHAT_ID, msg, reply_markup=kb)
                        notified[zid] = True
                        save_notified(notified)
                    else:
                        price = get_price(zone["pair"])
                        if price and not (float(zone["low"]) <= price <= float(zone["high"])):
                            if notified.get(zid):
                                notified.pop(zid,None)
                                save_notified(notified)
                except Exception as e:
                    logging.error(f"Zone {zid}: {e}")
        except Exception as e:
            logging.error(f"Surveillance: {e}")
        time.sleep(60)

# ─── MENUS ───────────────────────────────────
def main_menu():
    kb = InlineKeyboardMarkup()
    kb.row(InlineKeyboardButton("➕ Nouvelle Zone OB",   callback_data="new_zone"))
    kb.row(InlineKeyboardButton("📋 Mes zones actives",  callback_data="list_zones"))
    kb.row(InlineKeyboardButton("❌ Supprimer une zone", callback_data="delete_zone"))
    return kb

@bot.message_handler(commands=["start"])
def cmd_start(msg):
    user_state.pop(msg.chat.id,None)
    bot.send_message(msg.chat.id,"🤖 Bot OB Signal\nQue veux-tu faire ?",reply_markup=main_menu())

@bot.callback_query_handler(func=lambda c: True)
def handle_callback(call):
    cid  = call.message.chat.id
    mid  = call.message.message_id
    data = call.data
    bot.answer_callback_query(call.id)

    if data == "back_start":
        user_state.pop(cid,None)
        bot.edit_message_text("🤖 Bot OB Signal\nQue veux-tu faire ?",cid,mid,reply_markup=main_menu())

    elif data == "new_zone":
        user_state[cid] = {}
        kb = InlineKeyboardMarkup()
        kb.row(InlineKeyboardButton("EURUSD",callback_data="pair_EURUSD"),
               InlineKeyboardButton("USDJPY",callback_data="pair_USDJPY"))
        kb.row(InlineKeyboardButton("GBPUSD",callback_data="pair_GBPUSD"),
               InlineKeyboardButton("XAUUSD",callback_data="pair_XAUUSD"))
        kb.row(InlineKeyboardButton("GBPJPY",callback_data="pair_GBPJPY"),
               InlineKeyboardButton("AUDUSD",callback_data="pair_AUDUSD"))
        kb.row(InlineKeyboardButton("BTCUSD",callback_data="pair_BTCUSD"),
               InlineKeyboardButton("ETHUSD",callback_data="pair_ETHUSD"))
        kb.row(InlineKeyboardButton("Autre ✏️",callback_data="pair_OTHER"))
        bot.edit_message_text("📊 Quelle paire ?",cid,mid,reply_markup=kb)

    elif data.startswith("pair_"):
        pair = data.replace("pair_","")
        if pair == "OTHER":
            user_state[cid] = {"step":"pair_custom"}
            bot.edit_message_text("✏️ Tape le nom de ta paire (ex: USDCAD) :",cid,mid)
        else:
            user_state[cid] = {"pair":pair}
            kb = InlineKeyboardMarkup()
            kb.row(InlineKeyboardButton("🟢 BUY",callback_data="dir_BUY"),
                   InlineKeyboardButton("🔴 SELL",callback_data="dir_SELL"))
            bot.edit_message_text(f"✅ Paire : {pair}\n\nDirection ?",cid,mid,reply_markup=kb)

    elif data.startswith("dir_"):
        user_state[cid]["direction"] = data.replace("dir_","")
        kb = InlineKeyboardMarkup()
        kb.row(InlineKeyboardButton("H4",callback_data="tf_H4"),
               InlineKeyboardButton("H1",callback_data="tf_H1"),
               InlineKeyboardButton("M30",callback_data="tf_M30"))
        bot.edit_message_text(
            f"✅ {user_state[cid]['pair']} | {user_state[cid]['direction']}\n\nTimeframe de l'OB ?",
            cid,mid,reply_markup=kb)

    elif data.startswith("tf_"):
        user_state[cid]["timeframe"] = data.replace("tf_","")
        user_state[cid]["step"] = "price_high"
        bot.edit_message_text(
            f"✅ {user_state[cid]['pair']} | {user_state[cid]['direction']} | {user_state[cid]['timeframe']}\n\n"
            "📈 Entre le prix HAUT de ta zone OB :\n(copie-colle depuis TradingView)",cid,mid)

    elif data == "list_zones":
        zones = load_zones()
        if not zones:
            bot.edit_message_text("📋 Aucune zone active.",cid,mid)
            return
        lines = ["📋 Tes zones actives :\n"]
        for i,z in enumerate(zones,1):
            e = "🟢" if z["direction"]=="BUY" else "🔴"
            lines.append(f"{i}. {e} {z['pair']} — {z['direction']} — {z['timeframe']}\n"
                         f"   Haut: {z['high']} | Bas: {z['low']}\n"
                         f"   Créée: {z.get('created','—')}")
        kb = InlineKeyboardMarkup()
        kb.row(InlineKeyboardButton("🔙 Retour",callback_data="back_start"))
        bot.edit_message_text("\n".join(lines),cid,mid,reply_markup=kb)

    elif data == "delete_zone":
        zones = load_zones()
        if not zones:
            bot.edit_message_text("📋 Aucune zone à supprimer.",cid,mid)
            return
        kb = InlineKeyboardMarkup()
        for z in zones:
            e = "🟢" if z["direction"]=="BUY" else "🔴"
            kb.row(InlineKeyboardButton(
                f"{e} {z['pair']} {z['direction']} {z['timeframe']}",
                callback_data=f"del_{z['id']}"))
        kb.row(InlineKeyboardButton("🔙 Retour",callback_data="back_start"))
        bot.edit_message_text("❌ Quelle zone supprimer ?",cid,mid,reply_markup=kb)

    elif data.startswith("del_"):
        zid      = data.replace("del_","")
        zones    = [z for z in load_zones() if z["id"]!=zid]
        save_zones(zones)
        notified = load_notified()
        notified.pop(zid,None)
        save_notified(notified)
        bot.edit_message_text("✅ Zone supprimée !",cid,mid)

    elif data.startswith("enter_"):
        bot.edit_message_reply_markup(cid,mid)
        bot.send_message(cid,"✅ Trade pris ! Bonne chance 🎯")

    elif data.startswith("skip_"):
        bot.edit_message_reply_markup(cid,mid)
        bot.send_message(cid,"❌ Signal ignoré. On attend le prochain 👀")

@bot.message_handler(func=lambda m: True)
def handle_text(msg):
    cid  = msg.chat.id
    text = msg.text.strip()
    st   = user_state.get(cid,{})

    if st.get("step") == "pair_custom":
        user_state[cid] = {"pair":text.upper()}
        kb = InlineKeyboardMarkup()
        kb.row(InlineKeyboardButton("🟢 BUY",callback_data="dir_BUY"),
               InlineKeyboardButton("🔴 SELL",callback_data="dir_SELL"))
        bot.send_message(cid,f"✅ Paire : {text.upper()}\n\nDirection ?",reply_markup=kb)

    elif st.get("step") == "price_high":
        try:
            user_state[cid]["high"] = float(text.replace(",","."))
            user_state[cid]["step"] = "price_low"
            bot.send_message(cid,f"✅ Haut : {user_state[cid]['high']}\n\n📉 Entre le prix BAS de ta zone OB :")
        except:
            bot.send_message(cid,"❌ Prix invalide. Entre un nombre ex: 1.2550")

    elif st.get("step") == "price_low":
        try:
            low  = float(text.replace(",","."))
            high = user_state[cid]["high"]
            if low >= high:
                bot.send_message(cid,"❌ Le BAS doit être inférieur au HAUT. Réessaie :")
                return
            zones   = load_zones()
            zone_id = f"{st['pair']}_{st['direction']}_{st['timeframe']}_{int(time.time())}"
            zone    = {
                "id":zone_id,"pair":st["pair"],"direction":st["direction"],
                "timeframe":st["timeframe"],"high":high,"low":low,
                "created":datetime.now().strftime("%d/%m/%Y %H:%M"),
            }
            zones.append(zone)
            save_zones(zones)
            user_state.pop(cid,None)
            emoji = "🟢" if zone["direction"]=="BUY" else "🔴"
            bot.send_message(cid,
                f"✅ Zone enregistrée !\n\n"
                f"{emoji} {zone['pair']} — {zone['direction']} — OB {zone['timeframe']}\n"
                f"Haut : {high} | Bas : {low}\n\n"
                f"🟢 Surveillance active 24h/24"
            )
        except:
            bot.send_message(cid,"❌ Prix invalide. Entre un nombre ex: 1.2500")

if __name__ == "__main__":
    logging.info("🤖 Bot OB Signal v3 démarré !")
    t = threading.Thread(target=surveillance_loop, daemon=True)
    t.start()
    bot.infinity_polling()
