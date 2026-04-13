import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import google.generativeai as genai
import numpy as np
import streamlit as st
import xgboost as xgb
import yfinance as yf
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from datetime import date, datetime, timezone

# LLM_PROVIDER=gemini | ollama (if unset: use Gemini when GEMINI_API_KEY is set, else Ollama)
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-1.5-flash")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def _news_assessment_prompt(news_text: str) -> str:
    return f"""You are a financial analyst. Assess whether the following financial news is likely accurate or misleading.

News:
{news_text}

Respond in exactly this format on the first line:
VERDICT: ACCURATE
or
VERDICT: INACCURATE
or
VERDICT: UNCERTAIN

Then give brief supporting reasoning in 2-4 sentences. Do not give personalized investment advice."""


def _parse_verdict(text: str) -> Tuple[Optional[int], str]:
    first = text.split("\n", 1)[0].upper()
    prediction: Optional[int]
    if "VERDICT: ACCURATE" in first or first.startswith("ACCURATE"):
        prediction = 1
    elif "VERDICT: INACCURATE" in first or "INACCURATE" in first:
        prediction = 0
    elif "UNCERTAIN" in first:
        prediction = None
    else:
        if first.startswith("A") and "INACCURATE" not in first:
            prediction = 1
        elif "INACCURATE" in text.upper()[:80]:
            prediction = 0
        else:
            prediction = None
    return prediction, text


def _gemini_api_key() -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()
    try:
        return st.secrets.get("GEMINI_API_KEY", "").strip() or None
    except Exception:
        return None


def _default_llm_provider() -> str:
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit in ("gemini", "ollama"):
        return explicit
    return "gemini" if _gemini_api_key() else "ollama"


def _generative_model_compatible(name: str) -> bool:
    """Exclude models that only work with Interactions API despite list_models metadata."""
    n = name.lower()
    if "deep-research" in n:
        return False
    return True


def _preferred_model_index(models: list[str], preferred: str) -> int:
    if preferred in models:
        return models.index(preferred)
    for i, m in enumerate(models):
        if "gemini" in m.lower():
            return i
    return 0


@st.cache_data(ttl=3600)
def _list_gemini_models(api_key: str) -> list[str]:
    genai.configure(api_key=api_key)
    models: list[str] = []
    for m in genai.list_models():
        methods = set(getattr(m, "supported_generation_methods", []) or [])
        if "generateContent" in methods:
            name = getattr(m, "name", None)
            if name and _generative_model_compatible(str(name)):
                models.append(str(name))
    models.sort()
    return models


@st.cache_data(ttl=300)
def _list_ollama_models(base_url: str) -> list[str]:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read().decode())
        names = [str(m.get("name", "")).strip() for m in data.get("models", []) if m.get("name")]
        return sorted(set(names))
    except Exception:
        return []


def query_ollama(news_text: str, base_url: str, model_name: str) -> Tuple[Optional[int], str]:
    """Local Ollama — no API key. Requires `ollama serve` and a pulled model (e.g. ollama pull llama3.2)."""
    prompt = _news_assessment_prompt(news_text)
    url = f"{base_url.rstrip('/')}/api/generate"
    body = json.dumps({"model": model_name, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode(errors="replace")
        except Exception:
            pass
        return None, f"Ollama HTTP {e.code}: {e.reason}. {err_body[:200]}. Model `{model_name}` installed? Try: ollama pull {model_name}"
    except urllib.error.URLError as e:
        reason = str(e).lower()
        if "10061" in str(e) or "actively refused" in reason or "connection refused" in reason:
            hint = (
                "No Ollama server on this PC. Install Ollama from https://ollama.com/download "
                f"(or: winget install Ollama.Ollama), launch **Ollama** from the Start menu, then open a terminal and run: "
                f"`ollama pull {model_name}`."
            )
        else:
            hint = (
                f"Start Ollama (Start menu on Windows, or `ollama serve` in a terminal), then `ollama pull {model_name}`."
            )
        return None, f"Ollama: could not reach {base_url} ({e!s}). {hint}"
    except Exception as e:
        return None, f"Ollama request failed: {e!s}"

    text = (data.get("response") or "").strip()
    if not text:
        return None, "Ollama returned an empty response. Try a different model or prompt again."
    prediction, text_out = _parse_verdict(text)
    return prediction, text_out


def query_gemini(news_text: str, model_name: str) -> Tuple[Optional[int], str]:
    """Use Gemini to assess financial news. Returns (1=likely accurate, 0=likely misleading, None=uncertain), full text."""
    api_key = _gemini_api_key()
    if not api_key:
        return (
            None,
            "Add your Gemini API key: put GEMINI_API_KEY in a .env file next to app.py, set the env var, "
            "use .streamlit/secrets.toml, or the sidebar. Get a key at https://aistudio.google.com/apikey",
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = _news_assessment_prompt(news_text)

    try:
        response = model.generate_content(prompt)
    except Exception as e:
        return None, f"Gemini request failed: {e!s}. Check your API key and model name ({model_name})."

    text = (response.text or "").strip()
    if not text:
        return None, "Gemini returned an empty response. Try again or check your API key and quota."

    prediction, text_out = _parse_verdict(text)
    return prediction, text_out


# Streamlit UI
_backend_labels = ["Ollama (local, no API key)", "Gemini (Google AI Studio)"]
_default_backend = _default_llm_provider()
_backend_index = 0 if _default_backend == "ollama" else 1
llm_backend = st.sidebar.radio(
    "News analysis backend",
    _backend_labels,
    index=_backend_index,
    help="Ollama runs on your machine — https://ollama.com — no cloud API key.",
)
use_ollama = llm_backend.startswith("Ollama")

ollama_url = DEFAULT_OLLAMA_URL
ollama_model = DEFAULT_OLLAMA_MODEL
selected_model = DEFAULT_GEMINI_MODEL

if use_ollama:
    st.sidebar.header("Ollama")
    ollama_url = st.sidebar.text_input(
        "Ollama base URL",
        value=DEFAULT_OLLAMA_URL,
        help="Default: http://127.0.0.1:11434 — requires `ollama serve`",
    ).strip().rstrip("/") or DEFAULT_OLLAMA_URL
    _ollama_list = _list_ollama_models(ollama_url)
    if _ollama_list:
        _om_pref = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()
        _om_i = _ollama_list.index(_om_pref) if _om_pref in _ollama_list else 0
        ollama_model = st.sidebar.selectbox("Ollama model", options=_ollama_list, index=_om_i)
    else:
        ollama_model = st.sidebar.text_input(
            "Ollama model name",
            value=DEFAULT_OLLAMA_MODEL,
            help="Could not list models from Ollama. Example: llama3.2 — run: ollama pull llama3.2",
        ).strip() or DEFAULT_OLLAMA_MODEL
else:
    st.sidebar.header("Gemini (Google AI Studio)")
    sidebar_key = st.sidebar.text_input(
        "Gemini API key (optional if set in .env or GEMINI_API_KEY)",
        type="password",
        help="Free key: https://aistudio.google.com/apikey",
    )
    if sidebar_key:
        os.environ["GEMINI_API_KEY"] = sidebar_key

    api_key_now = _gemini_api_key()
    available_models: list[str] = []
    model_help = "Free key: https://aistudio.google.com/apikey"
    if api_key_now:
        try:
            available_models = _list_gemini_models(api_key_now)
            if not available_models:
                model_help = "No generateContent models found for this key."
        except Exception as e:
            model_help = f"Could not list models: {e!s}"

    _model_options = available_models if available_models else [DEFAULT_GEMINI_MODEL]
    selected_model = st.sidebar.selectbox(
        "Gemini model",
        options=_model_options,
        index=_preferred_model_index(_model_options, DEFAULT_GEMINI_MODEL),
        help=model_help,
    )

_analysis_name = "Ollama" if use_ollama else "Gemini"
st.title("Financial Misinformation Detector")
st.write(
    f"Enter financial news below. The app uses **{_analysis_name}** "
    f"({'local LLM via Ollama' if use_ollama else 'Google AI Studio'}) to assess whether it looks accurate or misleading."
)

input_news = st.text_area("Enter the financial news you want to check:")

if st.button("Check Accuracy"):
    if input_news:
        if use_ollama:
            prediction, evidence = query_ollama(input_news, ollama_url, ollama_model)
        else:
            prediction, evidence = query_gemini(input_news, selected_model)

        _config_error = evidence.startswith("Add your Gemini") or evidence.startswith("Ollama: could not reach")
        _http_error = evidence.startswith("Ollama HTTP")
        _request_failed = evidence.startswith("Ollama request failed") or evidence.startswith("Gemini request failed")

        if prediction is not None:
            if prediction == 1:
                st.success("The news is likely accurate.")
                st.subheader(f"Analysis ({_analysis_name}):")
                st.write(evidence)
            else:
                st.error("The news is likely misleading or inaccurate.")
                st.subheader(f"Analysis ({_analysis_name}):")
                st.write(evidence)
        else:
            if _config_error or _http_error or _request_failed:
                st.error(evidence)
            else:
                st.warning("Uncertain — verify with other sources.")
                st.subheader(f"Analysis ({_analysis_name}):")
                st.write(evidence)
    else:
        st.warning("Please enter some news text for evaluation.")

st.title("Stock Market Predictor")
st.write("Provide the ticker symbol for price prediction.")
input_ticker = st.text_area("Enter the stock ticker (e.g. AAPL, MSFT):")


def stocks(stockName):
    ticker = stockName.strip().upper()
    stock = yf.Ticker(ticker)
    first_trade_epoch = stock.info.get("firstTradeDateEpoch", None)
    start_date = "2010-01-01"
    if first_trade_epoch:
        start_date = datetime.fromtimestamp(first_trade_epoch, tz=timezone.utc).strftime("%Y-%m-%d")
    end_date = date.today()
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(stock_data["Close"].values.reshape(-1, 1))

    def create_dataset(data, lag=1):
        X, y = [], []
        for i in range(len(data) - lag - 1):
            X.append(data[i : (i + lag), 0])
            y.append(data[i + lag, 0])
        return np.array(X), np.array(y)

    lag = 5
    X, y = create_dataset(scaled_close, lag)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")

    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

    y_pred_lstm = model.predict(X_test)
    y_pred_lstm_2d = y_pred_lstm[:, -1, :]

    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    xgb_model.fit(y_pred_lstm_2d, y_test)

    y_pred_boosted = xgb_model.predict(y_pred_lstm_2d)

    y_pred_boosted_actual = scaler.inverse_transform(y_pred_boosted.reshape(-1, 1))

    predicted_closing_price = y_pred_boosted_actual[-1][0]
    return predicted_closing_price


predictedValue = 0

if st.button("Check Stock Market Prediction"):
    if input_ticker:
        predictedValue = stocks(input_ticker)

        if predictedValue != 0:
            st.write("The stock will close at the following predicted value: ")
            st.success(predictedValue)
            st.warning(
                "The product is in testing phase; do your own research. This is not financial advice."
            )

    else:
        st.warning("Please enter the correct stock ticker.")
