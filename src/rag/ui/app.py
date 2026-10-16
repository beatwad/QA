"""Streamlit GUI for the RAG API: log in as a known user, ask questions, sync the index (admins only).

Run: uv run streamlit run src/rag/ui/app.py   (the API must be up: uv run rag serve)
"""

import json
import os

import requests
import streamlit as st

API = os.environ.get("RAG_API_URL", "http://127.0.0.1:8080")
ADMIN_GROUP = "admin"


def sse_events(resp: requests.Response):
    kind = None
    for line in resp.iter_lines(decode_unicode=True):
        if line.startswith("event: "):
            kind = line.removeprefix("event: ")
        elif line.startswith("data: "):
            yield kind, json.loads(line.removeprefix("data: "))


def login(user: str) -> None:
    try:
        resp = requests.get(f"{API}/v1/me", headers={"X-User": user}, timeout=10)
    except requests.ConnectionError:
        st.error(f"Cannot reach the API at {API}. Is `rag serve` running?")
        return
    if resp.status_code != 200:
        st.error("Unknown user.")
        return
    st.session_state.update(user=user, groups=resp.json()["groups"], messages=[], conversation_id=None)
    st.rerun()


def render_citations(citations: list[dict]) -> None:
    for c in citations:
        with st.expander(f"[{c['n']}] {c['title']}"):
            st.write(c["snippet"])
            if c.get("url"):
                st.markdown(f"[{c['url']}]({c['url']})")


def ask(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        box, text, result = st.empty(), "", {}
        try:
            with requests.post(f"{API}/v1/ask", stream=True, timeout=300,
                               headers={"X-User": st.session_state.user},
                               json={"question": question, "conversation_id": st.session_state.conversation_id}) as resp:
                resp.raise_for_status()
                for kind, data in sse_events(resp):
                    if kind == "meta":
                        st.session_state.conversation_id = data["conversation_id"]
                    elif kind == "token":
                        text += data["text"]
                        box.write(text)
                    elif kind == "result":
                        result = data
        except requests.RequestException as e:
            box.error(f"Request failed: {e}")
            return
        answer = result.get("answer", text)  # the final answer may differ from the streamed text
        box.write(answer)
        render_citations(result.get("citations", []))
    st.session_state.messages.append({"role": "assistant", "content": answer, "citations": result.get("citations", [])})


def sync_panel() -> None:
    if st.sidebar.button("Sync database"):
        with st.sidebar, st.spinner("Syncing..."):
            resp = requests.post(f"{API}/v1/admin/sync", headers={"X-User": st.session_state.user}, timeout=3600)
        if resp.ok:
            st.sidebar.success(", ".join(f"{k}: {v}" for k, v in resp.json().items()))
        else:
            st.sidebar.error(f"Sync failed ({resp.status_code}): {resp.text}")


st.set_page_config(page_title="RAG Q&A bot")
st.markdown(
    "<style>[data-testid='stBottom'] > div {padding-bottom: 3rem;}</style>",
    unsafe_allow_html=True,
)

if "user" not in st.session_state:
    st.title("Log in")
    with st.form("login"):
        name = st.text_input("User")
        if st.form_submit_button("Log in") and name:
            login(name.strip())
    st.stop()

with st.sidebar:
    st.write(f"Logged in as **{st.session_state.user}**")
    st.caption(", ".join(st.session_state.groups))
    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()
    if st.button("New conversation"):
        st.session_state.update(messages=[], conversation_id=None)
        st.rerun()
if ADMIN_GROUP in st.session_state.groups:
    sync_panel()

st.title("RAG Q&A bot")
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        render_citations(m.get("citations", []))
if question := st.chat_input("Ask a question"):
    ask(question)
