import os, json, asyncio, logging
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes, ConversationHandler
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TOKEN     = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")
ZONES_FILE = "zones.json"
NOTIFIED_FILE = "notified.json"

logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# ÉTATS CONVERSATION
# ─────────────────────────────────────────────
PAIR, DIRECTION, TIMEFRAME, PRICE_HIGH, PRICE_LOW = range(5)

# ─────────────────────────────────────────────
# GESTION DES ZONES (fichier JSON)
# ─────────────────────────────────────────────
def load_zones():
    if os.path.exists(ZONES_FILE):
        with open(ZONES_FILE) as f:
            return json.load(f)
    return []

def save_zones(zones):
    with open(ZONES_FILE, "w") as f:
        json.dump(zones, f, indent=2)

def load_notified():
    if os.path.exists(NOTIFIED_FILE):
        with open(NOTIFIED_FILE) as f:
            return json.load(f)
    return {}

def save_notified(n):
    with open(NOTIFIED_FILE, "w") as f:
        json.dump(n, f, indent=2)

# ─────────────────────────────────────────────
# PAIRES → TICKER YFINANCE
# ─────────────────────────────────────────────
TICKER_MAP = {
    "EURUSD": "EURUSD=X", "USDJPY": "USDJPY=X",
    "GBPUSD": "GBPUSD=X", "XAUUSD": "GC=F",
    "GBPJPY": "GBPJPY=X", "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X",
    "BTCUSD": "BTC-USD",  "ETHUSD": "ETH-USD",
    "NAS100": "NQ=F",     "US30":   "YM=F",
}

def get_ticker(pair):
    return TICKER_MAP.get(pair.upper(), pair.upper() + "=X")

# ─────────────────────────────────────────────
# DONNÉES OHLCV
# ─────────────────────────────────────────────
TF_INTERVAL = {
    "H4": "1h", "H1": "60m", "M30": "30m",
    "M15": "15m", "M5": "5m", "M1": "1m",
}
TF_PERIOD = {
    "H4": "60d", "H1": "30d", "M30": "10d",
    "M15": "5d", "M5": "2d", "M1": "1d",
}
SUB_TFS = {
    "H4": ["H1", "M30", "M15"],
    "H1": ["M30", "M15", "M5"],
    "M30": ["M15", "M5", "M1"],
}

def get_data(pair, tf):
    try:
        interval = TF_INTERVAL[tf]
        period   = TF_PERIOD[tf]
        df = yf.download(get_ticker(pair), interval=interval,
                         period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty or len(df) < 52:
            return None
        if tf == "H4":
            df = df.resample("4h").agg(
                {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
            ).dropna()
        return df
    except Exception as e:
        logging.warning(f"Données indisponibles {pair} {tf}: {e}")
        return None

def get_current_price(pair):
    try:
        df = yf.download(get_ticker(pair), interval="1m", period="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return float(df["Close"].iloc[-1]) if not df.empty else None
    except:
        return None

# ─────────────────────────────────────────────
# CALCUL INDICATEURS
# ─────────────────────────────────────────────
def calc_rsi(closes, period=14):
    delta = closes.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def rsi_signal(val):
    if val < 30:   return 1   # BUY  (survente → rebond)
    if val <= 50:  return -1  # SELL
    if val <= 70:  return 1   # BUY
    return -1                 # SELL (surachat → retournement)

def ma10_signal(open_p, close_p, ma):
    body_top = max(open_p, close_p)
    body_bot = min(open_p, close_p)
    if body_bot > ma:  return 1   # corps entier au dessus → BUY
    if body_top < ma:  return -1  # corps entier en dessous → SELL
    return 0                      # corps touche → NEUTRE

def ichimoku_signal(close_p, sa, sb):
    if pd.isna(sa) or pd.isna(sb): return 0
    top = max(sa, sb)
    bot = min(sa, sb)
    if close_p > top: return 1
    if close_p < bot: return -1
    return 0

def get_indicators(pair, tf):
    df = get_data(pair, tf)
    if df is None or len(df) < 52:
        return None

    closes = df["Close"]
    rsi_s  = rsi_signal(float(calc_rsi(closes).iloc[-1]))

    ma10   = float(closes.rolling(10).mean().iloc[-1])
    ma_s   = ma10_signal(float(df["Open"].iloc[-1]),
                         float(df["Close"].iloc[-1]), ma10)

    # Ichimoku
    tenkan = (df["High"].rolling(9).max()  + df["Low"].rolling(9).min())  / 2
    kijun  = (df["High"].rolling(26).max() + df["Low"].rolling(26).min()) / 2
    sa     = ((tenkan + kijun) / 2).shift(26)
    sb     = ((df["High"].rolling(52).max() + df["Low"].rolling(52).min()) / 2).shift(26)
    ich_s  = ichimoku_signal(float(closes.iloc[-1]),
                             float(sa.iloc[-1]), float(sb.iloc[-1]))

    return {"rsi": rsi_s, "ma": ma_s, "ich": ich_s}

# ─────────────────────────────────────────────
# DÉTECTIONS SUPPLÉMENTAIRES
# ─────────────────────────────────────────────
def detect_inside_bar(pair, tf):
    df = get_data(pair, tf)
    if df is None or len(df) < 2: return False
    return (df["High"].iloc[-1] < df["High"].iloc[-2] and
            df["Low"].iloc[-1]  > df["Low"].iloc[-2])

def detect_sl_hunt(pair, tf):
    df = get_data(pair, tf)
    if df is None or len(df) < 11: return False
    ph = df["High"].iloc[-11:-1].max()
    pl = df["Low"].iloc[-11:-1].min()
    up   = df["High"].iloc[-1] > ph and df["Close"].iloc[-1] < ph
    down = df["Low"].iloc[-1]  < pl and df["Close"].iloc[-1] > pl
    return bool(up or down)

def detect_divergence(pair, tf):
    df = get_data(pair, tf)
    if df is None or len(df) < 6: return False
    price_up = float(df["Close"].iloc[-1]) > float(df["Close"].iloc[-6])
    vol_up   = float(df["Volume"].iloc[-1]) > float(df["Volume"].iloc[-6])
    return bool((price_up and not vol_up) or (not price_up and not vol_up))

# ─────────────────────────────────────────────
# ANALYSE COMPLÈTE D'UNE ZONE
# ─────────────────────────────────────────────
def analyze_zone(zone):
    pair      = zone["pair"]
    direction = zone["direction"]
    ob_tf     = zone["timeframe"]
    high      = zone["high"]
    low       = zone["low"]

    price = get_current_price(pair)
    if price is None:
        return None

    # Prix pas dans la zone → pas de signal
    if not (low <= price <= high):
        return None

    # Analyse timeframes inférieurs
    sub_tfs = SUB_TFS.get(ob_tf, ["M30", "M15", "M5"])
    dv       = 1 if direction == "BUY" else -1
    results  = {}
    confirmed = 0

    for tf in sub_tfs:
        ind = get_indicators(pair, tf)
        if ind is None:
            results[tf] = None
            continue
        count = sum(1 for k in ["rsi","ma","ich"] if ind[k] == dv)
        ok    = count >= 2
        if ok: confirmed += 1
        results[tf] = {**ind, "confirmed": ok, "count": count}

    if confirmed < 2:
        return None  # Pas assez de confirmation

    return {
        "pair":       pair,
        "direction":  direction,
        "ob_tf":      ob_tf,
        "price":      price,
        "results":    results,
        "inside_bar": detect_inside_bar(pair, ob_tf),
        "sl_hunt":    detect_sl_hunt(pair, ob_tf),
        "divergence": detect_divergence(pair, ob_tf),
    }

# ─────────────────────────────────────────────
# FORMATAGE DU MESSAGE DE SIGNAL
# ─────────────────────────────────────────────
def format_signal(sig):
    emoji = "🟢" if sig["direction"] == "BUY" else "🔴"
    dv    = 1 if sig["direction"] == "BUY" else -1

    lines = [
        f"{emoji} *SIGNAL {sig['direction']} — {sig['pair']} — OB {sig['ob_tf']}*",
        "━━━━━━━━━━━━━━━━━━",
        "*Analyse Timeframes :*",
    ]

    for tf, r in sig["results"].items():
        if r is None:
            lines.append(f"  {tf}: ❓ Données indisponibles")
            continue
        rsi_e = "✅" if r["rsi"] == dv else "❌"
        ma_e  = "✅" if r["ma"]  == dv else "❌"
        ich_e = "✅" if r["ich"] == dv else "❌"
        ok_e  = "✅ VALIDÉ" if r["confirmed"] else "❌"
        lines.append(f"  {tf}: RSI{rsi_e} MA10{ma_e} Ichi{ich_e} → {ok_e}")

    lines += [
        "━━━━━━━━━━━━━━━━━━",
        f"🔍 Chasse au SL  : {'✅ Détectée' if sig['sl_hunt']    else '❌ Absente'}",
        f"📦 Inside Bar    : {'✅ Présente' if sig['inside_bar'] else '❌ Absente'}",
        f"📊 Divergence    : {'⚠️ Divergence' if sig['divergence'] else '✅ Convergence'}",
        "━━━━━━━━━━━━━━━━━━",
        f"💰 Prix actuel   : `{sig['price']:.5f}`",
        "⚠️ _Place ton SL et TP manuellement_",
    ]
    return "\n".join(lines)

# ─────────────────────────────────────────────
# SURVEILLANCE EN ARRIÈRE-PLAN
# ─────────────────────────────────────────────
async def surveillance(app):
    await asyncio.sleep(10)
    logging.info("Surveillance démarrée ✅")
    while True:
        try:
            zones    = load_zones()
            notified = load_notified()

            for zone in zones:
                zid = zone.get("id", "")
                sig = analyze_zone(zone)

                if sig:
                    if notified.get(zid):
                        continue  # Déjà notifié pour cette entrée
                    msg = format_signal(sig)
                    keyboard = [[
                        InlineKeyboardButton("✅ J'entre",  callback_data=f"enter_{zid}"),
                        InlineKeyboardButton("❌ Je passe", callback_data=f"skip_{zid}"),
                    ]]
                    await app.bot.send_message(
                        chat_id=CHAT_ID,
                        text=msg,
                        parse_mode="Markdown",
                        reply_markup=InlineKeyboardMarkup(keyboard)
                    )
                    notified[zid] = True
                    save_notified(notified)
                else:
                    # Prix sorti de la zone → reset pour prochaine entrée
                    price = get_current_price(zone["pair"])
                    if price and not (zone["low"] <= price <= zone["high"]):
                        if notified.get(zid):
                            notified.pop(zid, None)
                            save_notified(notified)

        except Exception as e:
            logging.error(f"Erreur surveillance: {e}")

        await asyncio.sleep(60)  # Vérifie chaque minute

# ─────────────────────────────────────────────
# COMMANDES TELEGRAM
# ─────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("➕ Nouvelle Zone OB", callback_data="new_zone")],
        [InlineKeyboardButton("📋 Mes zones actives", callback_data="list_zones")],
        [InlineKeyboardButton("❌ Supprimer une zone", callback_data="delete_zone")],
    ]
    await update.message.reply_text(
        "🤖 *Bot OB Signal*\nQue veux-tu faire ?",
        reply_markup=InlineKeyboardMarkup(kb),
        parse_mode="Markdown"
    )

# ── NOUVELLE ZONE : étape 1 → Paire ──────────
async def new_zone(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    kb = [
        [InlineKeyboardButton("EURUSD", callback_data="pair_EURUSD"),
         InlineKeyboardButton("USDJPY", callback_data="pair_USDJPY")],
        [InlineKeyboardButton("GBPUSD", callback_data="pair_GBPUSD"),
         InlineKeyboardButton("XAUUSD", callback_data="pair_XAUUSD")],
        [InlineKeyboardButton("GBPJPY", callback_data="pair_GBPJPY"),
         InlineKeyboardButton("AUDUSD", callback_data="pair_AUDUSD")],
        [InlineKeyboardButton("BTCUSD", callback_data="pair_BTCUSD"),
         InlineKeyboardButton("Autre ✏️", callback_data="pair_OTHER")],
    ]
    await query.edit_message_text("📊 *Quelle paire ?*",
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode="Markdown")
    return PAIR

# ── Étape 2 → Direction ──────────────────────
async def set_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    pair = query.data.replace("pair_", "")
    if pair == "OTHER":
        context.user_data["step"] = "pair_custom"
        await query.edit_message_text("✏️ Tape le nom de ta paire (ex: USDCAD) :")
        return PAIR
    context.user_data["pair"] = pair
    kb = [[
        InlineKeyboardButton("🟢 BUY",  callback_data="dir_BUY"),
        InlineKeyboardButton("🔴 SELL", callback_data="dir_SELL"),
    ]]
    await query.edit_message_text(f"✅ Paire : *{pair}*\n\nDirection ?",
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode="Markdown")
    return DIRECTION

async def set_pair_custom(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("step") == "pair_custom":
        context.user_data["pair"] = update.message.text.upper().strip()
        context.user_data.pop("step")
        kb = [[
            InlineKeyboardButton("🟢 BUY",  callback_data="dir_BUY"),
            InlineKeyboardButton("🔴 SELL", callback_data="dir_SELL"),
        ]]
        await update.message.reply_text(
            f"✅ Paire : *{context.user_data['pair']}*\n\nDirection ?",
            reply_markup=InlineKeyboardMarkup(kb),
            parse_mode="Markdown"
        )
        return DIRECTION
    return PAIR

# ── Étape 3 → Timeframe ──────────────────────
async def set_direction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data["direction"] = query.data.replace("dir_", "")
    kb = [[
        InlineKeyboardButton("H4",  callback_data="tf_H4"),
        InlineKeyboardButton("H1",  callback_data="tf_H1"),
        InlineKeyboardButton("M30", callback_data="tf_M30"),
    ]]
    await query.edit_message_text(
        f"✅ Paire : *{context.user_data['pair']}* | Direction : *{context.user_data['direction']}*\n\nTimeframe de l'OB ?",
        reply_markup=InlineKeyboardMarkup(kb),
        parse_mode="Markdown"
    )
    return TIMEFRAME

# ── Étape 4 → Prix HAUT ──────────────────────
async def set_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data["timeframe"] = query.data.replace("tf_", "")
    await query.edit_message_text(
        f"✅ *{context.user_data['pair']}* | *{context.user_data['direction']}* | *{context.user_data['timeframe']}*\n\n"
        "📈 Entre le prix *HAUT* de ta zone OB :\n_(copie-colle depuis TradingView)_",
        parse_mode="Markdown"
    )
    return PRICE_HIGH

# ── Étape 5 → Prix BAS ───────────────────────
async def set_price_high(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        context.user_data["high"] = float(update.message.text.strip().replace(",", "."))
    except:
        await update.message.reply_text("❌ Prix invalide, entre un nombre (ex: 1.2550)")
        return PRICE_HIGH
    await update.message.reply_text(
        f"✅ Haut : *{context.user_data['high']}*\n\n"
        "📉 Entre le prix *BAS* de ta zone OB :",
        parse_mode="Markdown"
    )
    return PRICE_LOW

# ── Étape finale → Enregistrement ────────────
async def set_price_low(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        low = float(update.message.text.strip().replace(",", "."))
    except:
        await update.message.reply_text("❌ Prix invalide, entre un nombre (ex: 1.2500)")
        return PRICE_LOW

    high = context.user_data["high"]
    if low >= high:
        await update.message.reply_text("❌ Le BAS doit être inférieur au HAUT. Réessaie :")
        return PRICE_LOW

    zones = load_zones()
    zone_id = f"{context.user_data['pair']}_{context.user_data['direction']}_{context.user_data['timeframe']}_{int(datetime.now().timestamp())}"
    zone = {
        "id":        zone_id,
        "pair":      context.user_data["pair"],
        "direction": context.user_data["direction"],
        "timeframe": context.user_data["timeframe"],
        "high":      high,
        "low":       low,
        "created":   datetime.now().strftime("%d/%m/%Y %H:%M"),
    }
    zones.append(zone)
    save_zones(zones)

    emoji = "🟢" if zone["direction"] == "BUY" else "🔴"
    await update.message.reply_text(
        f"✅ *Zone enregistrée !*\n\n"
        f"{emoji} *{zone['pair']}* — {zone['direction']} — OB {zone['timeframe']}\n"
        f"Haut : `{high}` | Bas : `{low}`\n\n"
        f"🟢 Surveillance active 24h/24",
        parse_mode="Markdown"
    )
    context.user_data.clear()
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text("❌ Annulé.")
    return ConversationHandler.END

# ── LISTE DES ZONES ───────────────────────────
async def list_zones(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    zones = load_zones()
    if not zones:
        await query.edit_message_text("📋 Aucune zone active pour le moment.")
        return
    lines = ["📋 *Tes zones actives :*\n"]
    for i, z in enumerate(zones, 1):
        e = "🟢" if z["direction"] == "BUY" else "🔴"
        lines.append(
            f"{i}. {e} *{z['pair']}* — {z['direction']} — {z['timeframe']}\n"
            f"   Haut: `{z['high']}` | Bas: `{z['low']}`\n"
            f"   Créée: {z.get('created','—')}"
        )
    await query.edit_message_text("\n".join(lines), parse_mode="Markdown")

# ── SUPPRIMER UNE ZONE ────────────────────────
async def delete_zone_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    zones = load_zones()
    if not zones:
        await query.edit_message_text("📋 Aucune zone à supprimer.")
        return
    kb = []
    for z in zones:
        e = "🟢" if z["direction"] == "BUY" else "🔴"
        label = f"{e} {z['pair']} {z['direction']} {z['timeframe']}"
        kb.append([InlineKeyboardButton(label, callback_data=f"del_{z['id']}")])
    kb.append([InlineKeyboardButton("🔙 Retour", callback_data="back_start")])
    await query.edit_message_text("❌ *Quelle zone supprimer ?*",
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode="Markdown")

async def delete_zone_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    zid   = query.data.replace("del_", "")
    zones = load_zones()
    zones = [z for z in zones if z["id"] != zid]
    save_zones(zones)
    notified = load_notified()
    notified.pop(zid, None)
    save_notified(notified)
    await query.edit_message_text("✅ Zone supprimée !")

# ── RÉPONSE AUX BOUTONS SIGNAL ────────────────
async def handle_trade_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action, zid = query.data.split("_", 1)
    if action == "enter":
        await query.edit_message_reply_markup(None)
        await query.message.reply_text("✅ *Trade pris !* Bonne chance 🎯", parse_mode="Markdown")
    else:
        await query.edit_message_reply_markup(None)
        await query.message.reply_text("❌ *Signal ignoré.* On attend le prochain 👀", parse_mode="Markdown")

async def back_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    kb = [
        [InlineKeyboardButton("➕ Nouvelle Zone OB",  callback_data="new_zone")],
        [InlineKeyboardButton("📋 Mes zones actives", callback_data="list_zones")],
        [InlineKeyboardButton("❌ Supprimer une zone", callback_data="delete_zone")],
    ]
    await query.edit_message_text("🤖 *Bot OB Signal*\nQue veux-tu faire ?",
                                  reply_markup=InlineKeyboardMarkup(kb),
                                  parse_mode="Markdown")

# ─────────────────────────────────────────────
# LANCEMENT DU BOT
# ─────────────────────────────────────────────
def main():
    app = Application.builder().token(TOKEN).build()

    # Conversation pour ajouter une zone
    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(new_zone, pattern="^new_zone$")],
        states={
            PAIR: [
                CallbackQueryHandler(set_pair, pattern="^pair_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, set_pair_custom),
            ],
            DIRECTION: [CallbackQueryHandler(set_direction, pattern="^dir_")],
            TIMEFRAME: [CallbackQueryHandler(set_timeframe, pattern="^tf_")],
            PRICE_HIGH: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_price_high)],
            PRICE_LOW:  [MessageHandler(filters.TEXT & ~filters.COMMAND, set_price_low)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv)
    app.add_handler(CallbackQueryHandler(list_zones,          pattern="^list_zones$"))
    app.add_handler(CallbackQueryHandler(delete_zone_menu,    pattern="^delete_zone$"))
    app.add_handler(CallbackQueryHandler(delete_zone_confirm, pattern="^del_"))
    app.add_handler(CallbackQueryHandler(back_start,          pattern="^back_start$"))
    app.add_handler(CallbackQueryHandler(handle_trade_response, pattern="^(enter|skip)_"))

    # Lancer la surveillance en arrière-plan
    loop = asyncio.get_event_loop()
    loop.create_task(surveillance(app))

    print("🤖 Bot démarré !")
    app.run_polling()

if __name__ == "__main__":
    main()
