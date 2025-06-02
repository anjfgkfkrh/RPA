import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import time
import requests
import feedparser
import urllib.parse
import re 

from pypfopt import expected_returns, risk_models, EfficientFrontier, CLA, plotting
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("종합 주식 분석 및 포트폴리오 최적화 앱 📊📰")

if 'name_to_symbol_cache' not in st.session_state:
    st.session_state.name_to_symbol_cache = {}

def get_ticker_from_name(stock_name_query):
    if stock_name_query in st.session_state.name_to_symbol_cache:
        return st.session_state.name_to_symbol_cache[stock_name_query]
    search_url = f"https://query1.finance.yahoo.com/v1/finance/search?q={stock_name_query}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        search_data = response.json()
        if search_data.get('quotes'):
            best_quote = next((q for q in search_data['quotes'] if q.get('quoteType') == 'EQUITY' and q.get('symbol')), None)
            if not best_quote and search_data['quotes'] and search_data['quotes'][0].get('symbol'):
                best_quote = search_data['quotes'][0]
            if best_quote:
                symbol = best_quote['symbol']
                name_from_api = best_quote.get('longname', best_quote.get('shortname', stock_name_query))
                result = {'symbol': symbol, 'resolved_name': name_from_api, 'query_name': stock_name_query}
                st.session_state.name_to_symbol_cache[stock_name_query] = result
                return result
        return None
    except requests.exceptions.Timeout: st.sidebar.warning(f"'{stock_name_query}' 티커 검색 시간 초과."); return None
    except requests.exceptions.RequestException as e: st.sidebar.warning(f"'{stock_name_query}' 티커 검색 중 네트워크 오류: {e}"); return None
    except ValueError: st.sidebar.warning(f"'{stock_name_query}' 티커 검색 결과 파싱 실패."); return None

def fetch_stock_news(stock_display_name, stock_ticker, force_korean_news=False, num_articles=5):
    news_items = []
    query_terms = [term for term in [stock_display_name, stock_ticker] if term]
    query = " ".join(query_terms)
    if not query_terms: 
        query = "주식 뉴스" 
    else:
        query += " stock" 

    encoded_query = urllib.parse.quote_plus(query)

    if force_korean_news:
        lang_code, country_code = "ko", "KR"
    elif ".KS" in stock_ticker.upper() or ".KQ" in stock_ticker.upper():
        lang_code, country_code = "ko", "KR"
    else:
        lang_code, country_code = "en", "US"
    
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl={lang_code}&gl={country_code}&ceid={country_code}:{lang_code.split('-')[0]}"
    
    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:num_articles]:
            title = entry.get("title", "제목 없음")
            link = entry.get("link", "#")
            published_time_struct = entry.get("published_parsed")
            published_date = time.strftime("%Y년 %m월 %d일 %H:%M", published_time_struct) if published_time_struct else "날짜 정보 없음"
            
            summary = entry.get("summary", "요약 정보 없음")
            if summary and summary != "요약 정보 없음":
                summary = re.sub(r'<[^>]+>', '', summary).strip() 

            news_items.append({
                "title": title, "link": link, "published": published_date, "summary": summary
            })
    except Exception as e:
        st.sidebar.warning(f"'{stock_display_name}' 뉴스 피드({lang_code}/{country_code}) 조회 중 오류: {e}")
    return news_items

st.sidebar.header("종목 및 옵션 선택")
stock_name_input_string = st.sidebar.text_input(
    "주식 종목 이름을 입력하세요 (쉼표로 구분)", "Apple,Microsoft,Samsung Electronics,Google,Nvidia"
)


st.sidebar.subheader("그래프 옵션") 
graph_options_config = {
    "1분 (1일)": {"fetch_args": {"period": "1d", "interval": "1m"}, "description_suffix": " (1분, 1일)"},
    "1시간 (2주)": {"fetch_args": {"period": "14d", "interval": "1h"}, "description_suffix": " (1시간, 2주)"},
    "1일 (180일)": {"fetch_args": {"period": "180d", "interval": "1d"}, "description_suffix": " (1일, 180일)"},
}
selected_graph_label = st.sidebar.selectbox("그래프 표시 기간:", list(graph_options_config.keys()))
current_graph_option = graph_options_config[selected_graph_label]

st.sidebar.subheader("뉴스 검색 옵션") 
search_korean_news_only = st.sidebar.checkbox("모든 종목에 대해 한국 뉴스로 검색", False)

st.sidebar.subheader("기타 옵션") 
auto_refresh = st.sidebar.checkbox("1분마다 자동 갱신 활성화", False)
refresh_interval_seconds = 60
risk_free_rate_input = st.sidebar.number_input("무위험 수익률 (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f") / 100

resolved_stocks_info, failed_resolutions = [], []
if stock_name_input_string:
    input_names = [name.strip() for name in stock_name_input_string.split(',') if name.strip()]
    if input_names:
        with st.spinner("종목 코드를 조회 중입니다..."):
            for name_query in input_names:
                stock_info = get_ticker_from_name(name_query)
                if stock_info: resolved_stocks_info.append(stock_info)
                else: failed_resolutions.append(name_query)
else: input_names = []

if not resolved_stocks_info and not input_names:
    st.info("왼쪽 사이드바에서 조회할 주식의 이름을 하나 이상 입력하세요.")
elif not resolved_stocks_info and input_names:
    st.error(f"입력하신 모든 종목의 코드를 찾을 수 없습니다: {', '.join(input_names)}")
else:
    if failed_resolutions:
        st.warning(f"다음 종목은 코드를 찾지 못했습니다: {', '.join(failed_resolutions)}")

    if resolved_stocks_info:
        tab_titles = [info['resolved_name'] for info in resolved_stocks_info]
        try: tab_objects = st.tabs(tab_titles)
        except Exception as e: st.error(f"탭 생성 중 오류: {e}. ({tab_titles})"); st.stop()

        for i, stock_info_item in enumerate(resolved_stocks_info):
            current_ticker_symbol = stock_info_item['symbol']
            current_display_name = stock_info_item['resolved_name']

            with tab_objects[i]:
                st.subheader(f"{current_display_name} ({current_ticker_symbol}) - {current_graph_option['description_suffix']}")
                try:
                    data = yf.download(current_ticker_symbol, period=current_graph_option["fetch_args"]["period"], interval=current_graph_option["fetch_args"]["interval"], progress=False, auto_adjust=False)
                    if not data.empty:
                        y_column_identifier = None
                        if isinstance(data.columns, pd.MultiIndex):
                            direct_match_col = ('Close', current_ticker_symbol); upper_match_col = ('Close', current_ticker_symbol.upper())
                            if direct_match_col in data.columns: y_column_identifier = direct_match_col
                            elif upper_match_col in data.columns: y_column_identifier = upper_match_col
                            else: fallback_col = next((c for c in data.columns if isinstance(c, tuple) and c and c[0] == 'Close'), None); y_column_identifier = fallback_col
                        elif 'Close' in data.columns: y_column_identifier = 'Close'
                        
                        if y_column_identifier and y_column_identifier in data:
                            actual_y_series_for_graph = data[y_column_identifier]
                            fig_graph = px.line(x=data.index, y=actual_y_series_for_graph, title=f"{current_display_name} 시세")
                            fig_graph.update_layout(xaxis_title="시간", yaxis_title="종가", showlegend=True, xaxis_fixedrange=True, yaxis_fixedrange=True)
                            fig_graph.update_traces(name=current_ticker_symbol, selector=dict(type='scatter'))
                            st.plotly_chart(fig_graph, use_container_width=True)
                        else: st.warning(f"'{current_display_name}': 시세 데이터에서 '종가' 정보를 찾을 수 없습니다.")
                    else: st.warning(f"'{current_display_name} ({current_ticker_symbol})' 시세 데이터를 가져올 수 없습니다.")
                except Exception as e: st.error(f"'{current_display_name} ({current_ticker_symbol})' 그래프 처리 중 오류: {type(e).__name__} - {e}")
                
                st.markdown("---")
                st.subheader(f"📰 최신 뉴스 - {current_display_name}")
                
                news_list = fetch_stock_news(current_display_name, current_ticker_symbol, 
                                             force_korean_news=search_korean_news_only,
                                             num_articles=5)
                
                if not news_list:
                    st.info("관련 뉴스를 찾을 수 없거나, 뉴스 피드를 가져오는 데 실패했습니다.")
                else:
                    for item_idx, item in enumerate(news_list):
                        try: source_domain = urllib.parse.urlparse(item['link']).netloc; source_display = f"출처: {source_domain}"
                        except: source_display = "출처 정보 없음"
                        st.markdown(f"""
                        <div style="margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #e0e0e0;">
                            <h5 style="margin-bottom: 5px;">
                                <a href="{item['link']}" target="_blank" style="text-decoration: none; color: #1a0dab;">
                                    {item['title']}
                                </a>
                            </h5>
                            <p style="font-size: 0.85em; color: #006621; margin-bottom: 5px;">
                                {source_display} | {item['published']}
                            </p>
                            <p style="font-size: 0.9em; color: #545454; margin-bottom: 0;">
                                {item['summary'][:250]}{'...' if len(item['summary']) > 250 else ''}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("💼 포트폴리오 최적화")
    if not resolved_stocks_info or len(resolved_stocks_info) < 2:
        st.info("포트폴리오 최적화를 실행하려면 최소 2개 이상의 유효한 종목이 필요합니다.")
    else:
        if st.button("포트폴리오 최적화 및 효율적 투자선 보기"):
            with st.spinner("최적화용 데이터를 조회하고 포트폴리오를 계산 중입니다..."):
                opt_period = "2y"; price_data_for_opt = pd.DataFrame()
                for stock_info in resolved_stocks_info:
                    ticker = stock_info['symbol']
                    try:
                        df_ticker_data = yf.download(ticker, period=opt_period, interval="1d", progress=False, auto_adjust=True)
                        if isinstance(df_ticker_data, pd.DataFrame) and not df_ticker_data.empty:
                            if 'Close' in df_ticker_data.columns:
                                price_data_candidate = df_ticker_data['Close']; actual_price_series_for_opt = None
                                if isinstance(price_data_candidate, pd.Series): actual_price_series_for_opt = price_data_candidate
                                elif isinstance(price_data_candidate, pd.DataFrame) and not price_data_candidate.empty and len(price_data_candidate.columns) == 1:
                                    actual_price_series_for_opt = price_data_candidate.iloc[:, 0]
                                if actual_price_series_for_opt is not None and isinstance(actual_price_series_for_opt, pd.Series):
                                    if actual_price_series_for_opt.count() > len(actual_price_series_for_opt) * 0.8: price_data_for_opt[ticker] = actual_price_series_for_opt
                                    else: st.warning(f"'{stock_info['resolved_name']}({ticker})': 유효 데이터 부족으로 제외 (유효 {actual_price_series_for_opt.count()}/{len(actual_price_series_for_opt)})")
                                else: st.warning(f"'{stock_info['resolved_name']}({ticker})': 'Close' Series 추출 실패 (타입: {type(price_data_candidate)})")
                            else: st.warning(f"'{stock_info['resolved_name']}({ticker})': 'Close' 컬럼 없음 (컬럼: {df_ticker_data.columns if isinstance(df_ticker_data, pd.DataFrame) else 'N/A'})")
                        elif isinstance(df_ticker_data, pd.DataFrame) and df_ticker_data.empty: st.warning(f"'{stock_info['resolved_name']}({ticker})': 데이터 없음 (결과 비어있음).")
                        else: st.warning(f"'{stock_info['resolved_name']}({ticker})': 예상치 못한 데이터 타입: {type(df_ticker_data)}")
                    except Exception as e: st.warning(f"'{stock_info['resolved_name']}({ticker})' 데이터 조회 오류 ({type(e).__name__}): {e}")
                if not price_data_for_opt.empty: price_data_for_opt.dropna(inplace=True)
                if len(price_data_for_opt.columns) < 2: st.error("최적화를 위한 유효 종목 데이터가 2개 미만입니다.")
                else:
                    try:
                        mu = expected_returns.mean_historical_return(price_data_for_opt); S = risk_models.sample_cov(price_data_for_opt)
                        ef = EfficientFrontier(mu, S); ef.max_sharpe(risk_free_rate=risk_free_rate_input); cleaned_weights = ef.clean_weights()
                        st.subheader("🎯 최적 포트폴리오 결과 (최대 샤프 지수 기준)")
                        weights_df = pd.DataFrame(list(cleaned_weights.items()), columns=['종목 티커', '비중'])
                        ticker_to_name_map = {info['symbol']: info['resolved_name'] for info in resolved_stocks_info}
                        weights_df['종목명'] = weights_df['종목 티커'].map(ticker_to_name_map)
                        weights_df = weights_df[['종목명', '종목 티커', '비중']].sort_values(by="비중", ascending=False)
                        weights_df['비중'] = weights_df['비중'].map('{:.2%}'.format); st.dataframe(weights_df.set_index('종목명'))
                        expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate_input)
                        st.markdown("---"); col1, col2, col3 = st.columns(3)
                        with col1: st.metric("예상 연간 수익률", f"{expected_annual_return:.2%}")
                        with col2: st.metric("예상 연간 변동성", f"{annual_volatility:.2%}")
                        with col3: st.metric("샤프 지수", f"{sharpe_ratio:.2f}")
                        st.caption(f"무위험 수익률 {risk_free_rate_input*100:.1f}% 가정. {opt_period}간의 일일 데이터를 기반으로 계산됨.")
                        st.subheader("📊 효율적 투자선 (Efficient Frontier)")
                        try:
                            cla = CLA(mu, S); cla.max_sharpe(); cla.min_volatility()
                            fig_ef, ax = plt.subplots(figsize=(10, 6))
                            plotting.plot_efficient_frontier(cla, ax=ax, show_assets=True, show_tickers=True, show_gmv=True, show_max_sharpe=True, risk_free_rate=risk_free_rate_input)
                            st.pyplot(fig_ef); plt.close(fig_ef)
                        except ImportError: st.warning("Matplotlib 미설치로 그래프 표시 불가. `pip install matplotlib`로 설치.")
                        except Exception as e_plot: st.error(f"효율적 투자선 그래프 생성 오류: {type(e_plot).__name__} - {e_plot}")
                    except Exception as e_opt: st.error(f"포트폴리오 최적화 오류: {type(e_opt).__name__} - {e_opt}")

if auto_refresh and resolved_stocks_info:
    time.sleep(refresh_interval_seconds)
    st.rerun()