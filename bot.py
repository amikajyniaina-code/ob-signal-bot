import os, json, logging
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes, ConversationHandler
)

TOKEN      = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID    = os.environ.get("TELEGRAM_CHAT_ID")
ZONES_FILE    = "zones.json"
NOTIFIED_FILE = "notified.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

PAIR, DIRECTION, TIMEFRAME, PRICE_HIGH, PRICE_LOW = range(5)

# ─── ZONES ───────────────────────────────────
def load_zones():
    try:
        if os.path.exists(ZONES_FILE):
            with open(ZONES_FILE) as f:
                return json.load(f)
    except:
        pass
    return []

def save_zones(z):
    with open(ZONES_FILE, "w") as f:
        json.dump(z, f, indent=2)

def load_notified():
    try:
        if os.path.exists(NOTIFIED_FILE):
            with open(NOTIFIED_FILE) as f:
                return json.load(f)
    except:
        pass
    return {}

def save_notified(n):
    with open(NOTIFIED_FILE, "w") as f:
        json.dump(n, f, indent=2)

# ─── TICKERS ─────────────────────────────────
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

# ─── DONNÉES ─────────────────────────────────
TF_MAP = {
    "H4":  ("1h",  "60d"),
    "H1":  ("60m", "30d"),
    "M30": ("30m", "10d"),
    "M15": ("15m", "5d"),
    "M5":  ("5m",  "2d"),
    "M1":  ("1m",  "1d"),
}
SUB_TFS = {
    "H4":  ["H1",  "M30", "M15"],
    "H1":  ["M30", "M15", "M5"],
    "M30": ["M15", "M5",  "M1"],
}

def get_data(pair, tf):
    try:
        interval, period = TF_MAP[tf]
        df = yf.download(
            get_ticker(pair), interval=interval,
            period=period, progress=False, auto_adjust=True
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.dropna()
        if tf == "H4":
            df = df.resample("4h").agg({
                "Open":"first","High":"max",
                "Low":"min","Close":"last","Volume":"sum"
            }).dropna()
        if len(df) < 20:
            return None
        return df
    except Exception as e:
        logging.warning(f"get_data error {pair} {tf}: {e}")
        return None

def get_price(pair):
    # Essaie 1m d'abord, puis 5m si vide
    for interval, period in [("1m","1d"), ("5m","2d")]:
        try:
            df = yf.download(
                get_ticker(pair), interval=interval,
                period=period, progress=False, auto_adjust=True
            )
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if not df.empty:
                return float(df["Close"].iloc[-1])
        except:
            continue
    return None

# ─── INDICATEURS ─────────────────────────────
def calc_rsi(closes, period=14):
    delta = closes.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / (loss + 1e-10)
    rsi   = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def rsi_signal(val):
    if val < 30:   return 1   # BUY survente
    if val <= 50:  return -1  # SELL
    if val <= 70:  return 1   # BUY
    return -1                 # SELL surachat

def ma10_signal(open_p, close_p, ma):
    body_top = max(open_p, close_p)
    body_bot = min(open_p, close_p)
    if body_bot > ma:  return 1   # corps au dessus → BUY
    if body_top < ma:  return -1  # corps en dessous → SELL
    return 0                      # corps touche → NEUTRE

def ich_signal(df):
    try:
        hi = df["High"]
        lo = df["Low"]
        cl = df["Close"]
        t  = (hi.rolling(9).max()  + lo.rolling(9).min())  / 2
        k  = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
        sa = ((t + k) / 2).shift(26)
        sb = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
        sa_val = sa.dropna()
        sb_val = sb.dropna()
        if sa_val.empty or sb_val.empty:
            return 0
        top = max(float(sa_val.iloc[-1]), float(sb_val.iloc[-1]))
        bot = min(float(sa_val.iloc[-1]), float(sb_val.iloc[-1]))
        c   = float(cl.iloc[-1])
        if c > top:  return 1
        if c < bot:  return -1
        return 0
    except:
        return 0

def get_indicators(pair, tf):
    df = get_data(pair, tf)
    if df is None:
        logging.warning(f"Pas de données {pair} {tf}")
        return None
    try:
        rsi_val = calc_rsi(df["Close"])
        rsi     = rsi_signal(rsi_val)
        ma_val  = float(df["Close"].rolling(10).mean().iloc[-1])
        ma      = ma10_signal(
            float(df["Open"].iloc[-1]),
            float(df["Close"].iloc[-1]),
            ma_val
        )
        ich = ich_signal(df)
        logging.info(f"{pair} {tf} → RSI:{rsi_val:.1f}({rsi}) MA:{ma} ICH:{ich}")
        return {"rsi": rsi, "ma": ma, "ich": ich, "rsi_val": rsi_val}
    except Exception as e:
        logging.warning(f"get_indicators error {pair} {tf}: {e}")
        return None

# ─── DÉTECTIONS SUPPLÉMENTAIRES ──────────────
def detect_inside_bar(pair, tf):
    df = get_data(pair, tf)
    if df is None or len(df) < 2: return False
    return bool(
        float(df["High"].iloc[-1]) < float(df["High"].iloc[-2]) and
        float(df["Low"].iloc[-1])  > float(df["Low"].iloc[-2])
    )

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
    price_up = float(df["Close"].iloc[-1]) > float(df["Close"].iloc[-6])
    vol_up   = float(df["Volume"].iloc[-1]) > float(df["Volume"].iloc[-6])
    return bool((price_up and not vol_up) or (not price_up and not vol_up))

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

    logging.info(f"Prix {pair}: {price:.5f} | Zone: {low} - {high}")

    if not (low <= price <= high):
        logging.info(f"{pair} hors zone ({price:.5f} pas entre {low} et {high})")
        return None

    logging.info(f"✅ {pair} DANS la zone ! Analyse en cours...")

    sub_tfs   = SUB_TFS.get(ob_tf, ["M30", "M15", "M5"])
    dv        = 1 if direction == "BUY" else -1
    results   = {}
    confirmed = 0

    for tf in sub_tfs:
        ind = get_indicators(pair, tf)
        if ind is None:
            results[tf] = None
            continue

        score = 0
        # RSI toujours compté
        if ind["rsi"] == dv: score += 1
        # MA toujours compté sauf si neutre
        if ind["ma"] != 0:
            if ind["ma"] == dv: score += 1
        # Ichimoku compté sauf si neutre
        if ind["ich"] != 0:
            if ind["ich"] == dv: score += 1

        ok = (score >= 2)
        if ok: confirmed += 1

        results[tf] = {**ind, "confirmed": ok, "score": score}
        logging.info(f"  {tf}: score={score} ok={ok}")

    logging.info(f"Total confirmés: {confirmed}/3 (besoin 2)")

    if confirmed < 2:
        logging.info(f"❌ Signal insuffisant pour {pair} {direction}")
        return None

    logging.info(f"🚀 SIGNAL {direction} validé pour {pair}!")

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

# ─── MESSAGE ─────────────────────────────────
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
        ma_e  = "✅" if r["ma"]  == dv else ("⚪" if r["ma"] == 0 else "❌")
        ich_e = "✅" if r["ich"] == dv else ("⚪" if r["ich"] == 0 else "❌")
        ok_e  = "✅ VALIDÉ" if r["confirmed"] else "❌"
        rsi_v = f"{r.get('rsi_val',0):.1f}"
        lines.append(f"  {tf}: RSI{rsi_e}({rsi_v}) MA{ma_e} Ichi{ich_e} → {ok_e}")
    lines += [
        "━━━━━━━━━━━━━━━━━━",
        f"🔍 Chasse au SL  : {'✅ Détectée'   if sig['sl_hunt']    else '❌ Absente'}",
        f"📦 Inside Bar    : {'✅ Présente'   if sig['inside_bar'] else '❌ Absente'}",
        f"📊 Divergence    : {'⚠️ Divergence' if sig['divergence'] else '✅ Convergence'}",
        "━━━━━━━━━━━━━━━━━━",
        f"💰 Prix actuel   : `{sig['price']:.5f}`",
        "⚠️ _Place ton SL et TP manuellement_",
    ]
    return "\n".join(lines)

# ─── SURVEILLANCE (job_queue) ─────────────────
async def surveillance_job(context: ContextTypes.DEFAULT_TYPE):
    zones    = load_zones()
    notified = load_notified()
    logging.info(f"🔍 Vérification {len(zones)} zone(s)...")

    for zone in zones:
        zid = zone.get("id", "")
        try:
            sig = analyze_zone(zone)
            if sig:
                if notified.get(zid):
                    continue
                msg      = format_signal(sig)
                keyboard = [[
                    InlineKeyboardButton("✅ J'entre",  callback_data=f"enter_{zid}"),
                    InlineKeyboardButton("❌ Je passe", callback_data=f"skip_{zid}"),
                ]]
                await context.bot.send_message(
                    chat_id=CHAT_ID, text=msg,
                    parse_mode="Markdown",
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
                notified[zid] = True
                save_notified(notified)
            else:
                price = get_price(zone["pair"])
                if price and not (float(zone["low"]) <= price <= float(zone["high"])):
                    if notified.get(zid):
                        notified.pop(zid, None)
                        save_notified(notified)
        except Exception as e:
            logging.error(f"Erreur zone {zid}: {e}")

# ─── HANDLERS TELEGRAM ───────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("➕ Nouvelle Zone OB",   callback_data="new_zone")],
        [InlineKeyboardButton("📋 Mes zones actives",  callback_data="list_zones")],
        [InlineKeyboardButton("❌ Supprimer une zone", callback_data="delete_zone")],
    ]
    await update.message.reply_text(
        "🤖 *Bot OB Signal*\nQue veux-tu faire ?",
        reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown"
    )

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
         InlineKeyboardButton("ETHUSD", callback_data="pair_ETHUSD")],
        [InlineKeyboardButton("Autre ✏️", callback_data="pair_OTHER")],
    ]
    await query.edit_message_text(
        "📊 *Quelle paire ?*",
        reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown"
    )
    return PAIR

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
    await query.edit_message_text(
        f"✅ Paire : *{pair}*\n\nDirection ?",
        reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown"
    )
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
            reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown"
        )
        return DIRECTION
    return PAIR

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
        f"✅ *{context.user_data['pair']}* | *{context.user_data['direction']}*\n\nTimeframe de l'OB ?",
        reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown"
    )
    return TIMEFRAME

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

async def set_price_high(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        context.user_data["high"] = float(update.message.text.strip().replace(",", "."))
    except:
        await update.message.reply_text("❌ Prix invalide. Entre un nombre ex: 1.2550")
        return PRICE_HIGH
    await update.message.reply_text(
        f"✅ Haut : *{context.user_data['high']}*\n\n📉 Entre le prix *BAS* de ta zone OB :",
        parse_mode="Markdown"
    )
    return PRICE_LOW

async def set_price_low(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        low = float(update.message.text.strip().replace(",", "."))
    except:
        await update.message.reply_text("❌ Prix invalide. Entre un nombre ex: 1.2500")
        return PRICE_LOW
    high = context.user_data["high"]
    if low >= high:
        await update.message.reply_text("❌ Le BAS doit être inférieur au HAUT. Réessaie :")
        return PRICE_LOW
    zones   = load_zones()
    zone_id = f"{context.user_data['pair']}_{context.user_data['direction']}_{context.user_data['timeframe']}_{int(datetime.now().timestamp())}"
    zone    = {
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
            f"{i}\\. {e} *{z['pair']}* — {z['direction']} — {z['timeframe']}\n"
            f"   Haut: `{z['high']}` \\| Bas: `{z['low']}`\n"
            f"   Créée: {z.get('created','—')}"
        )
    await query.edit_message_text("\n".join(lines), parse_mode="MarkdownV2")

async def delete_zone_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    zones = load_zones()
    if not zones:
        await query.edit_message_text("📋 Aucune zone à supprimer.")
        return
    kb = []
    for z in zones:
        e     = "🟢" if z["direction"] == "BUY" else "🔴"
        label = f"{e} {z['pair']} {z['direction']} {z['timeframe']}"
        kb.append([InlineKeyboardButton(label, callback_data=f"del_{z['id']}")])
    kb.append([InlineKeyboardButton("🔙 Retour", callback_data="back_start")])
    await query.edit_message_text(
        "❌ *Quelle zone supprimer ?*",
        reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown"
    )

async def delete_zone_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    zid      = query.data.replace("del_", "")
    zones    = [z for z in load_zones() if z["id"] != zid]
    save_zones(zones)
    notified = load_notified()
    notified.pop(zid, None)
    save_notified(notified)
    await query.edit_message_text("✅ Zone supprimée !")

async def handle_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query  = update.callback_query
    await query.answer()
    action = query.data.split("_")[0]
    await query.edit_message_reply_markup(None)
    if action == "enter":
        await query.message.reply_text("✅ *Trade pris !* Bonne chance 🎯", parse_mode="Markdown")
    else:
        await query.message.reply_text("❌ *Signal ignoré.* On attend le prochain 👀", parse_mode="Markdown")

async def back_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    kb = [
        [InlineKeyboardButton("➕ Nouvelle Zone OB",   callback_data="new_zone")],
        [InlineKeyboardButton("📋 Mes zones actives",  callback_data="list_zones")],
        [InlineKeyboardButton("❌ Supprimer une zone", callback_data="delete_zone")],
    ]
    await query.edit_message_text(
        "🤖 *Bot OB Signal*\nQue veux-tu faire ?",
        reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown"
    )

# ─── MAIN ────────────────────────────────────
def main():
    app = Application.builder().token(TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(new_zone, pattern="^new_zone$")],
        states={
            PAIR:       [
                CallbackQueryHandler(set_pair, pattern="^pair_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, set_pair_custom),
            ],
            DIRECTION:  [CallbackQueryHandler(set_direction,  pattern="^dir_")],
            TIMEFRAME:  [CallbackQueryHandler(set_timeframe,  pattern="^tf_")],
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
    app.add_handler(CallbackQueryHandler(handle_trade,        pattern="^(enter|skip)_"))

    # Job queue — vérifie toutes les 60 secondes
    app.job_queue.run_repeating(surveillance_job, interval=60, first=10)

    print("🤖 Bot OB Signal démarré !")
    app.run_polling()

if __name__ == "__main__":
    main()

