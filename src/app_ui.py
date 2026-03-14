# src/app_ui.py
import json
import io
import contextlib
import streamlit as st

from main_graph import build_app  # 用你現有的 build_app()

st.set_page_config(page_title="RAG LangGraph Debug UI", layout="wide")

st.title("🧪 RAG LangGraph Debug UI")
st.caption("用來觀察：難易度判斷 → 檢索/擴展 → 生成 → 評估 → 是否擴檢索（rewrite loop）")

# -----------------------
# Sidebar：只放設定，不放問題輸入
# -----------------------
with st.sidebar:
    st.header("設定")
    max_rounds = st.slider("🔁 最大擴檢索輪數（max_rounds）", 0, 5, 3)
    show_console = st.checkbox("顯示 Console Log（print 輸出）", value=True)
    stream_mode = st.selectbox("stream_mode", ["values", "updates"], index=0)
    run_btn = st.button("🚀 執行", type="primary")

# -----------------------
# 主區塊：輸入區（你要移出 sidebar）
# -----------------------
st.subheader("📝 輸入")
product_id = st.text_input("🏷️ 產品代號/名稱", value="")
query = st.text_area("問題", height=120, value="")

st.divider()

# -----------------------
# Helpers
# -----------------------
def safe_json(obj):
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)

def get_confidence_from_eval(ev: dict) -> float:
    """優先抓 C_conf，其次信心分數/coverage_score"""
    if not isinstance(ev, dict):
        return 0.0
    for k in ["C_conf", "信心分數", "confidence", "coverage_score"]:
        v = ev.get(k, None)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            continue
    return 0.0

def render_sources(expanded_docs):
    st.subheader("📚 來源（expanded_docs 預覽）")
    if not expanded_docs:
        st.info("（沒有 expanded_docs）")
        return

    # 顯示前 10 筆避免太長
    for i, d in enumerate(expanded_docs[:10]):
        meta = (d.get("meta", {}) or {})
        title = meta.get("TITLE", "")
        cid = meta.get("CHUNK_ID", "")
        group = meta.get("GROUP", "")
        with st.expander(f"[{i}] CHUNK_ID={cid} | {title} | {group}".strip()):
            txt = d.get("text", "") or ""
            st.write(txt[:2000] + ("..." if len(txt) > 2000 else ""))

def render_core(state):
    st.subheader("🧠 狀態 / Query")
    cols = st.columns(3)
    cols[0].write(f"**difficulty**: {state.get('difficulty')}")
    cols[1].write(f"**query_original**: {state.get('query_original')}")
    cols[2].write(f"**query_current**: {state.get('query_current')}")

    if state.get("extra_context"):
        with st.expander("🧩 extra_context（rewrite 後補充線索）"):
            st.write(state.get("extra_context", ""))

    if state.get("rewrite_raw"):
        with st.expander("🛠️ rewrite_raw（query_expanding_metrics 原始輸出）"):
            st.write(state.get("rewrite_raw", ""))

    if state.get("logs"):
        with st.expander("📌 logs（_log 收集）"):
            st.code("\n".join(state["logs"]))

def render_answer(answer: str):
    st.subheader("✅ 答案")
    st.write(answer or "")

def render_eval(state):
    st.subheader("📏 評估（evaluate_answer_metrics）")
    ev = state.get("eval", {}) or {}
    raw = state.get("eval_raw", "") or ""

    if not ev and raw:
        st.info("（eval 解析不到 dict，先顯示 raw）")
        st.code(raw, language="json")
        return

    conf = get_confidence_from_eval(ev)
    must_failed = state.get("must_failed", False)
    round_ = state.get("round", 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("confidence (C_conf優先)", f"{conf:.4f}")
    c2.metric("must_failed", str(must_failed))
    c3.metric("round", str(round_))

    if ev:
        st.json(ev)

def extract_expanded_from_state(state):
    graded = state.get("graded", {}) or {}
    expanded = graded.get("expanded_docs", None)
    if isinstance(expanded, list) and expanded:
        return expanded
    expanded2 = state.get("expanded_docs", None)
    if isinstance(expanded2, list) and expanded2:
        return expanded2
    return []

# -----------------------
# Streaming run
# -----------------------
def run_graph_stream(product_id: str, query: str, max_rounds: int, stream_mode: str = "values"):
    app = build_app()
    init_state = {
        "product_id": product_id,
        "query_original": query,
        "max_rounds": max_rounds,
    }

    # 收集 print 輸出
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if stream_mode == "updates":
            # updates 會回傳每一步更新的片段（依 langgraph 版本可能是 dict of node->delta）
            for event in app.stream(init_state, stream_mode="updates"):
                yield event, None
        else:
            # values：每一步回傳「完整 state」
            for state in app.stream(init_state, stream_mode="values"):
                yield state, None

    console = buf.getvalue()
    yield None, console

# -----------------------
# Main action (上下區塊 + streaming)
# -----------------------
if run_btn:
    if not product_id.strip() or not query.strip():
        st.error("請先填 產品代號/名稱 和 問題")
    else:
        # 這些 placeholder 會被逐步更新（你要的一塊一塊出來）
        core_ph = st.empty()
        retrieve_ph = st.empty()
        answer_ph = st.empty()
        eval_ph = st.empty()
        console_ph = st.empty()
        final_state_ph = st.empty()

        latest_state = {}

        with st.status("執行中…（會逐步顯示每一步結果）", expanded=True) as status:
            for payload, console in run_graph_stream(product_id.strip(), query.strip(), max_rounds, stream_mode=stream_mode):
                if console is not None:
                    # 最後才會拿到 console
                    if show_console:
                        with console_ph.container():
                            with st.expander("🖥️ Console Log（print 的輸出）", expanded=False):
                                st.code(console)
                    break

                # stream_mode="values"：payload 是完整 state
                if stream_mode == "values":
                    state = payload or {}
                    latest_state = state

                    # 依照 state 目前有哪些欄位，逐步渲染
                    with core_ph.container():
                        render_core(state)

                    expanded_docs = extract_expanded_from_state(state)
                    with retrieve_ph.container():
                        render_sources(expanded_docs)

                    with answer_ph.container():
                        render_answer(state.get("answer", ""))

                    with eval_ph.container():
                        render_eval(state)

                else:
                    # stream_mode="updates"：payload 比較像「局部更新事件」
                    # 先把事件印出來，避免你看不到（之後你想要更漂亮再做 node-based UI）
                    with core_ph.container():
                        st.subheader("🧩 Stream updates（局部事件）")
                        st.json(payload)

                status.update(label="執行中…（更新中）", state="running")

            status.update(label="完成 ✅", state="complete")

        # 最後把 final state 展開（可選）
        if latest_state:
            with final_state_ph.container():
                with st.expander("🧾 Final State JSON", expanded=False):
                    st.code(safe_json(latest_state), language="json")