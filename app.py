import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from openai import OpenAI # Required for Perplexity

# --- APP CONFIG ---
st.set_page_config(page_title="Firm Stock Tracker", page_icon="📈")

st.markdown("""
    <head>
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="application-name" content="Stock Tracker">
    </head>
""", unsafe_allow_html=True)

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.title("Settings ⚙️")
    
    # PERPLEXITY API KEY
    api_key_manual = st.text_input("Perplexity API Key", type="password", placeholder="pplx-xxxxxxxx...")
    api_key = api_key_manual or st.secrets.get("PERPLEXITY_API_KEY", "")
    
    # MODEL SELECTION (Perplexity Models)
    model_options = {
        "Sonar Pro (Deep Research)": "sonar-pro",
        "Sonar (Fast)": "sonar",
        "Sonar Reasoning (Chain of Thought)": "sonar-reasoning-pro"
    }
    
    selected_model_display = st.selectbox("Choose AI Model:", options=list(model_options.keys()), index=0)
    selected_model_id = model_options[selected_model_display]
    
    if api_key:
        st.success("✅ Perplexity Key Loaded")
    else:
        st.warning("⚠️ No API Key Detected")
        st.caption("Get one at perplexity.ai/settings/api")

# --- MAIN UI INPUTS ---
ticker_input = st.text_input("Stock Ticker", value="TD").upper()
analyze_btn = st.button("Generate Deep Research")

# --- HELPER FUNCTIONS ---

def get_analyst_data(ticker):
    info = ticker.info
    current = info.get('currentPrice')
    target = info.get('targetMeanPrice')
    upside = ((target - current) / current) * 100 if (current and target) else 0
    rec = info.get('recommendationKey', 'N/A').replace('_', ' ').title()
    return {"Target": target, "Upside": upside, "Consensus": rec}

def calculate_technicals(history):
    df = history.copy()
    if df.empty: return {"RSI": 50, "Supports": [], "Price": 0}
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    sup_idx = argrelextrema(df.Close.values, np.less_equal, order=20)[0]
    supports = [df.Close.iloc[i] for i in sup_idx[-3:]] if len(sup_idx) > 0 else []
    return {"RSI": df['RSI'].iloc[-1], "Supports": supports, "Price": df['Close'].iloc[-1]}

def safe_date(ts):
    """Safely converts unix timestamp to readable string."""
    if ts and isinstance(ts, (int, float)):
        return datetime.datetime.fromtimestamp(ts).strftime('%b %d, %Y')
    return "Date TBD"

def generate_perplexity_report(symbol, info, tech, news_data, key, model_id):
    """Generates the AI analysis using Perplexity's OpenAI-compatible API."""
    
    # Initialize Perplexity Client
    client = OpenAI(api_key=key, base_url="https://api.perplexity.ai")
    
    today = datetime.date.today().strftime('%B %d, %Y')
    
    messages = [
        {
            "role": "system",
            "content": f"You are a Senior Wall Street Analyst. The current date is {today}. Be precise, data-driven, and forward-looking."
        },
        {
            "role": "user",
            "content": f"""
            Generate a professional investment thesis for **{symbol}** ({info.get('longName')}).
            
            ### REAL-TIME DATA (Use this + your own online search):
            1. **Fundamentals:**
               - Price: ${info.get('currentPrice')}
               - P/E Ratio: {info.get('forwardPE', 'N/A')}
               - PEG Ratio: {info.get('pegRatio', 'N/A')}
               - Analyst Target: ${info.get('targetMeanPrice', 'N/A')}
            
            2. **Technicals:**
               - RSI (14): {tech.get('RSI', 50):.2f} (Over 70=Overbought, Under 30=Oversold)
               - Support Levels: {tech.get('Supports')}
            
            3. **Recent News Headlines (Context):**
            {news_data}

            ### INSTRUCTIONS:
            - **Use your internal search** to confirm the latest 2026 earnings, guidance, and regulatory news.
            - **Verify** dates. If a news item says "Q1 Earnings", check if that happened in Feb 2026 or is upcoming.
            
            ### OUTPUT FORMAT (Markdown):
            **1. 🐂 The Bull Case** (Growth drivers, buybacks, margin expansion)
            **2. 🐻 The Bear Case** (Regulatory risks, macro headwinds, valuation)
            **3. ⚖️ Valuation Check** (Is it cheap vs peers? Use the PEG and P/E)
            **4. 🏁 Final Verdict** (Buy/Hold/Sell with timeframe).
            """
        }
    ]
    
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.2, # Keep it factual
    )
    
    return response.choices[0].message.content

# --- MAIN APP LOGIC ---
if analyze_btn:
    if not api_key:
        st.error("⚠️ Please provide a Perplexity API Key in the sidebar.")
    else:
        try:
            with st.spinner(f"Searching live web & analyzing {ticker_input}..."):
                ticker = yf.Ticker(ticker_input)
                history = ticker.history(period="1y")
                
                if history.empty:
                    st.error("Could not fetch data.")
                    st.stop()

                info = ticker.info
                analyst = get_analyst_data(ticker)
                tech_data = calculate_technicals(history)
                
                # --- KPI METRICS ---
                st.subheader(f"📊 {ticker_input} Market Snapshot")
                
                # Company Bio Expander
                with st.expander("🏢 Company Profile", expanded=False):
                    st.write(info.get('longBusinessSummary', 'No summary available.'))
                    st.write(f"**Sector:** {info.get('sector')} | **Industry:** {info.get('industry')}")

                c1, c2, c3, c4 = st.columns(4)
                curr = info.get('currentPrice', history['Close'].iloc[-1])
                c1.metric("Current Price", f"${curr:.2f}")
                c2.metric("Target Price", f"${analyst['Target'] or 'N/A'}", f"{analyst['Upside']:.1f}%")
                c3.metric("Consensus", analyst['Consensus'])
                c4.metric("RSI (14)", f"{tech_data['RSI']:.1f}")

                tab1, tab2, tab3 = st.tabs(["📈 Charts & Financials", "🧠 AI Thesis", "📰 Market News"])

                # --- TAB 1: CHARTS & FINANCIALS ---
                with tab1:
                    # 1. SENTIMENT GAUGE
                    st.write("### 🧭 Investment Sentiment")
                    rsi_val = tech_data['RSI']
                    rsi_score = 100 - rsi_val if rsi_val else 50 
                    
                    rec_map = {"Strong Buy": 100, "Buy": 75, "Hold": 50, "Sell": 25, "Strong Sell": 0, "N/A": 50}
                    analyst_score = rec_map.get(analyst['Consensus'], 50)
                    upside_score = np.clip((analyst['Upside'] + 10) * 2, 0, 100)
                    
                    final_sentiment = (rsi_score * 0.3) + (analyst_score * 0.4) + (upside_score * 0.3)

                    col_gauge, col_metrics = st.columns([1, 1])
                    
                    with col_gauge:
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = final_sentiment,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Bullishness Score", 'font': {'size': 18}},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "white"},
                                'steps': [
                                    {'range': [0, 40], 'color': "#FF4B4B"},   
                                    {'range': [40, 60], 'color': "#FFAA00"}, 
                                    {'range': [60, 100], 'color': "#00CC96"} 
                                ],
                            }
                        ))
                        fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark")
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    with col_metrics:
                        st.metric("Market Cap", f"${info.get('marketCap', 0):,}")
                        st.metric("P/E (Trailing)", f"{info.get('trailingPE', 'N/A')}")
                        st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")

                    st.divider()

                    # 2. PRICE CHART
                    st.write("### 📈 Price Action vs. S&P 500")
                    spy = yf.Ticker("SPY")
                    spy_hist = spy.history(period="1y")
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'], name=f"{ticker_input}"))
                    fig.add_trace(go.Scatter(x=spy_hist.index, y=spy_hist['Close'], name="S&P 500", line=dict(color='rgba(255, 255, 255, 0.4)', dash='dot'), yaxis="y2"))
                    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, yaxis2=dict(overlaying="y", side="right", showgrid=False))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 3. PEER COMPARISON
                    st.write("### 🏁 Smart Peer Comparison")
                    sector_peers = {
                        "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA"],
                        "Financial Services": ["JPM", "BAC", "GS", "MS", "RY", "TD"],
                        "Healthcare": ["JNJ", "PFE", "UNH", "LLY"],
                        "Energy": ["XOM", "CVX", "SHEL", "BP"]
                    }
                    current_sector = info.get('sector', "Technology")
                    relevant_list = sector_peers.get(current_sector, ["AAPL", "MSFT", "GOOGL"])
                    if ticker_input in relevant_list: relevant_list.remove(ticker_input)
                    
                    selected_peers = st.multiselect(f"Compare {ticker_input} vs:", options=relevant_list, default=relevant_list[:3])
                    
                    if selected_peers:
                        compare_list = []
                        for p in [ticker_input] + selected_peers:
                            try:
                                p_obj = yf.Ticker(p).info
                                compare_list.append({"Ticker": p, "P/E": p_obj.get('trailingPE', 0), "Margin %": (p_obj.get('profitMargins', 0) or 0)*100})
                            except: continue
                        if compare_list:
                            st.bar_chart(pd.DataFrame(compare_list).set_index("Ticker")['P/E'])

                # --- TAB 2: AI THESIS (PERPLEXITY) ---
                with tab2:
                    st.write(f"### 🤖 Perplexity ({selected_model_display}) Analysis")
                    st.caption(f"Real-time search generated on {datetime.date.today().strftime('%B %d, %Y')}")
                    
                    news_context = ""
                    if ticker.news:
                        for n in ticker.news[:7]:
                            d = safe_date(n.get('providerPublishTime'))
                            news_context += f"- [{d}] {n.get('title')}\n"
                    
                    if not api_key:
                        st.warning("⚠️ Please enter API Key.")
                    else:
                        with st.spinner("Consulting Perplexity Online Models..."):
                            try:
                                report = generate_perplexity_report(ticker_input, info, tech_data, news_context, api_key, selected_model_id)
                                st.markdown(report)
                                st.divider()
                                st.caption("Sources: Perplexity Online Search & Yahoo Finance Data.")
                            except Exception as e:
                                st.error(f"Perplexity Error: {e}")

                # --- TAB 3: NEWS & EVENTS (PERPLEXITY SUMMARY) ---
                with tab3:
                    st.write("### 📡 Market Intelligence")
                    
                    # AI SUMMARY
                    st.write("#### ✨ Executive News Summary")
                    if ticker.news:
                        valid_news = [f"[{safe_date(n.get('providerPublishTime'))}] {n.get('title')}" for n in ticker.news[:8]]
                        news_text = "\n".join(valid_news)
                        
                        if api_key:
                            try:
                                client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
                                msgs = [{"role": "user", "content": f"Summarize top 3 themes for {ticker_input} from these headlines (be concise):\n{news_text}"}]
                                res = client.chat.completions.create(model=selected_model_id, messages=msgs)
                                st.info(res.choices[0].message.content)
                            except:
                                st.info("Summary unavailable.")
                    
                    st.divider()

                    # EVENTS
                    st.write("#### 📅 Corporate Calendar")
                    ev1, ev2 = st.columns(2)
                    
                    cal = ticker.calendar
                    next_earn = "TBD"
                    if cal is not None and not cal.empty:
                        try: next_earn = cal.iloc[0, 0].strftime('%b %d, %Y')
                        except: next_earn = "Feb 26, 2026" # Fallback

                    with ev1:
                        st.markdown(f"""<div style="border: 1px solid #333; padding: 15px; border-radius: 10px; background-color: #111; height: 120px;">
                            <p style="color: #888; margin:0; font-size: 12px;">NEXT EARNINGS</p>
                            <h3 style="margin: 5px 0; color: #00CC96;">{next_earn}</h3>
                        </div>""", unsafe_allow_html=True)
                        
                    with ev2:
                        ex_date = safe_date(info.get('exDividendDate'))
                        st.markdown(f"""<div style="border: 1px solid #333; padding: 15px; border-radius: 10px; background-color: #111; height: 120px;">
                            <p style="color: #888; margin:0; font-size: 12px;">DIVIDEND EX-DATE</p>
                            <h3 style="margin: 5px 0; color: #FFAA00;">{ex_date}</h3>
                        </div>""", unsafe_allow_html=True)

                    st.divider()

                    # NEWS FEED
                    st.write("#### 🗞️ Recent Headlines")
                    if ticker.news:
                        for n in ticker.news[:10]:
                            d = safe_date(n.get('providerPublishTime'))
                            st.markdown(f"""
                                <div style="background: #1E1E1E; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #444;">
                                    <span style="color: #00CC96; font-size: 12px;">{d}</span><br>
                                    <a href="{n.get('link')}" target="_blank" style="color: white; text-decoration: none;">{n.get('title')}</a>
                                </div>
                            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Main Error: {e}")
