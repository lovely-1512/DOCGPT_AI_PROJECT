import os, sys, json, uuid, sqlite3, time, re
from datetime import datetime
import streamlit as st

st.set_page_config(page_title="DocGPT AI", page_icon="📚", layout="wide")
sys.path.append(os.path.dirname(__file__))
from rag_pipeline import create_engine
try: from groq import Groq
except Exception: Groq = None

CHAT_HISTORY_FILE, USER_DB = "chat_history.json", "docgpt_users.db"
GROQ_MODEL, WHISPER_MODEL = "openai/gpt-oss-120b", "whisper-large-v3-turbo"

def initialize_database():
    with sqlite3.connect(USER_DB) as c:
        c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, created_at TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS analytics (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                  "username TEXT, event TEXT, created_at TEXT)")
initialize_database()

def log_event(event):
    user = st.session_state.get("username", "guest")
    try:
        with sqlite3.connect(USER_DB) as c:
            c.execute("INSERT INTO analytics(username,event,created_at) VALUES (?,?,?)",
                      (user, event, datetime.now().isoformat()))
    except Exception: pass

def load_chat_history():
    if not os.path.exists(CHAT_HISTORY_FILE): return {}
    try:
        with open(CHAT_HISTORY_FILE, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception: return {}

def save_chat_history(chats):
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(chats, f, indent=2, ensure_ascii=False)
    except Exception: pass

def new_chat():
    return {"id": str(uuid.uuid4()), "title":"New Conversation", "messages":[],
            "created_at": datetime.now().isoformat(), "updated_at": datetime.now().isoformat()}

def make_title(msgs):
    for m in msgs:
        if m.get("role") == "user":
            w = m.get("content","").strip().split()
            return (" ".join(w[:7]) + "...") if len(w) > 7 else " ".join(w) or "New Conversation"
    return "New Conversation"

def save_current_chat():
    cid, msgs = st.session_state.current_chat_id, st.session_state.messages
    if not msgs: return
    prev = st.session_state.chat_history.get(cid, {})
    st.session_state.chat_history[cid] = {
        "id":cid, "title":make_title(msgs), "messages":msgs,
        "created_at":prev.get("created_at", datetime.now().isoformat()),
        "updated_at":datetime.now().isoformat()}
    save_chat_history(st.session_state.chat_history)

def load_chat(cid):
    c = st.session_state.chat_history.get(cid)
    if c:
        st.session_state.current_chat_id = cid
        st.session_state.messages = c.get("messages", [])

def delete_chat(cid):
    st.session_state.chat_history.pop(cid, None)
    save_chat_history(st.session_state.chat_history)
    st.session_state.current_chat_id = new_chat()["id"]
    st.session_state.messages = []

# ---------- session state ----------
for k, v in [("engine", None), ("messages", []), ("processed_files", []),
             ("mode", "Chat"), ("chat_history", None), ("current_chat_id", None),
             ("edit_index", None), ("answer_style", "Natural"), ("answer_length", "Medium"),
             ("show_analytics", False), ("username", "Guest")]:
    if k not in st.session_state:
        st.session_state[k] = create_engine() if k == "engine" else \
            load_chat_history() if k == "chat_history" else \
            new_chat()["id"] if k == "current_chat_id" else v

st.markdown("""<style>
.main-title{font-size:2.45rem;font-weight:800;margin-bottom:.05rem}
.subtitle{color:#777;font-size:.96rem;margin-bottom:1rem}
.history-heading{font-size:.8rem;font-weight:800;color:#777;margin:.5rem 0 .4rem;
  text-transform:uppercase;letter-spacing:.05em}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="main-title">📚 DocGPT AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your intelligent document workspace</div>', unsafe_allow_html=True)

# ---------- render helpers ----------
def render_ai_answer(c):
    st.markdown(c or "No answer generated.", unsafe_allow_html=False)

def render_confidence(conf, label="Grounding confidence"):
    if not conf: return
    try: s = float(conf.get("score", 0))
    except (TypeError, ValueError): s = 0.0
    st.caption(f"◉ {label}: {conf.get('label','Unknown')} • {s:.2f}")

def render_citations(cits):
    if not cits: return
    with st.expander("📚 Sources & Evidence", expanded=False):
        for i, c in enumerate(cits, 1):
            st.markdown(f"**{c.get('evidence_id', i)} · {c.get('source','Unknown')}**")
            st.caption(f"Location: {c.get('location','N/A')}")
            if i < len(cits): st.divider()

def render_evidence(ev, title="📚 Supporting Evidence"):
    if not ev: return
    with st.expander(title, expanded=False):
        for i, e in enumerate(ev, 1):
            st.markdown(f"**Evidence {i} · {e.get('source','Unknown')}**")
            st.caption(f"Location: {e.get('location','N/A')}")
            if e.get("text"): st.markdown(f"> {e['text']}")
            if i < len(ev): st.divider()

def style_instr():
    return {"Natural":"Answer naturally.","Simple":"Use very simple language.",
            "Detailed":"Deep explanation with examples.","Technical":"Technical accurate terms.",
            "Exam Answer":"Exam-ready structure.","Interview Answer":"Concise professional."}.get(
            st.session_state.answer_style, "Answer naturally.")

def length_instr():
    return {"Short":"2-5 sentences.","Medium":"Balanced short paragraphs.",
            "Detailed":"Comprehensive but focused."}.get(st.session_state.answer_length, "Balanced.")

# ---------- sidebar ----------
with st.sidebar:
    st.header("📚 DocGPT AI")
    if st.button("＋ New Chat", type="primary", use_container_width=True):
        save_current_chat(); st.session_state.current_chat_id = new_chat()["id"]
        st.session_state.messages = []; st.rerun()
    st.markdown('<div class="history-heading">💬 Previous Chats</div>', unsafe_allow_html=True)
    chats = sorted(st.session_state.chat_history.values(),
                   key=lambda x: x.get("updated_at",""), reverse=True)
    if not chats: st.caption("No previous chats yet.")
    for ch in chats:
        cid, title = ch.get("id"), ch.get("title","New Conversation")[:32]
        active = cid == st.session_state.current_chat_id
        if st.button(("● " if active else "  ") + title, key=f"open_{cid}", use_container_width=True):
            save_current_chat(); load_chat(cid); st.rerun()
        if active and st.button("🗑️ Delete this chat", key=f"del_{cid}", use_container_width=True):
            delete_chat(cid); st.rerun()
    st.divider(); st.subheader("🚀 Workspace")
    mode = st.radio("Mode", ["Chat","Compare","Study","Quiz"],
                    index=["Chat","Compare","Study","Quiz"].index(st.session_state.mode))
    st.session_state.mode = mode
    st.divider()
    if mode == "Chat":
        st.subheader("🎯 Answer Settings")
        st.session_state.answer_style = st.selectbox("Answer style",
            ["Natural","Simple","Detailed","Technical","Exam Answer","Interview Answer"],
            index=["Natural","Simple","Detailed","Technical","Exam Answer","Interview Answer"]
                  .index(st.session_state.answer_style))
        st.session_state.answer_length = st.select_slider("Answer length",
            options=["Short","Medium","Detailed"], value=st.session_state.answer_length)
        st.divider()
    st.subheader("📂 Documents")
    files = st.file_uploader("Upload files", type=["pdf","pptx","docx","txt","csv","xlsx"],
                             accept_multiple_files=True)
    if st.button("⚡ Process Documents", type="primary", use_container_width=True,
                 disabled=not files):
        os.makedirs("raw_data", exist_ok=True); paths = []
        for f in files:
            p = os.path.join("raw_data", os.path.basename(f.name))
            with open(p, "wb") as o: o.write(f.getbuffer())
            paths.append(p)
        with st.spinner("Analyzing, chunking and indexing..."):
            try: r = st.session_state.engine.build_knowledge_base(paths)
            except Exception as e: r = {"success":False, "message":str(e)}
        if r.get("success"):
            st.session_state.processed_files = r.get("documents", [])
            log_event("process_documents"); st.success(f"Indexed {r.get('chunks',0)} chunks.")
        else: st.error(r.get("message","Unable to process."))
    eng = st.session_state.engine
    if eng.ready:
        st.caption(f"🟢 {len(eng.documents)} docs • {eng.total_chunks} chunks")
        with st.expander("📄 Processed Documents"):
            for d in eng.documents: st.write(f"• {d}")
    else: st.info("Upload documents to begin.")
    st.divider()
    if mode == "Chat":
        st.subheader("🎙️ Voice Input")
        audio = st.audio_input("Record your question")
        if audio is not None and Groq:
            try:
                key = os.environ.get("GROQ_API_KEY")
                if key:
                    vc = Groq(api_key=key)
                    tr = vc.audio.transcriptions.create(
                        file=(audio.name, audio.getvalue()), model=WHISPER_MODEL)
                    st.session_state.voice_text = tr.text
                    st.success("Voice converted to text.")
            except Exception as e: st.error(f"Voice error: {e}")
        if st.session_state.get("voice_text"):
            st.text_area("Transcribed", value=st.session_state.voice_text, height=80, disabled=True)
            if st.button("💬 Ask this question", use_container_width=True):
                st.session_state.voice_prompt = st.session_state.voice_text
                st.session_state.voice_text = ""; st.rerun()
        st.divider()
    st.subheader("⚙️ Controls")
    if st.button("🧹 Clear Current Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history.pop(st.session_state.current_chat_id, None)
        save_chat_history(st.session_state.chat_history); st.rerun()
    if st.button("🔄 Refresh", use_container_width=True):
        save_current_chat(); st.rerun()
    if st.session_state.messages:
        st.subheader("⬇️ Export")
        transcript = "\n".join(
            f"{'USER' if m.get('role')=='user' else 'DOCGPT AI'}:\n{m.get('content','')}\n"
            for m in st.session_state.messages)
        st.download_button("📄 Download TXT", transcript, "DocGPT_Conversation.txt",
                           mime="text/plain", use_container_width=True)
        st.download_button("📝 Download Markdown", transcript, "DocGPT_Conversation.md",
                           mime="text/markdown", use_container_width=True)
    st.divider()
    if st.button("📊 Usage Analytics", use_container_width=True):
        st.session_state.show_analytics = not st.session_state.show_analytics; st.rerun()
    st.caption(f"👤 {st.session_state.username}")

voice_prompt = st.session_state.pop("voice_prompt", None)

# ---------- CHAT ----------
if mode == "Chat":
    st.subheader("💬 Chat with your Documents")
    st.caption("Ask naturally and get grounded answers from your uploaded documents.")
    for i, m in enumerate(st.session_state.messages):
        role, content = m.get("role","assistant"), m.get("content","")
        with st.chat_message(role):
            if role == "user":
                st.markdown(content)
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✏️ Edit", key=f"edit_{i}"):
                        st.session_state.edit_index = i; st.rerun()
                with c2:
                    if st.button("🗑️ Delete", key=f"del_msg_{i}"):
                        del st.session_state.messages[i]
                        if i < len(st.session_state.messages) and \
                           st.session_state.messages[i].get("role") == "assistant":
                            del st.session_state.messages[i]
                        save_current_chat(); st.rerun()
            else:
                render_ai_answer(content)
                render_citations(m.get("citations", []))
                render_confidence(m.get("confidence", {}))
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🔄 Regenerate", key=f"regen_{i}") and i > 0:
                        prev = st.session_state.messages[i-1]
                        if prev.get("role") == "user":
                            q = prev["content"]; del st.session_state.messages[i]
                            with st.spinner("Regenerating..."):
                                try: r = eng.ask_rag(q, conversation_history=st.session_state.messages)
                                except Exception: r = {"answer":"Failed.","citations":[],"confidence":{}}
                            st.session_state.messages.insert(i, {
                                "role":"assistant","content":r.get("answer",""),
                                "citations":r.get("citations",[]),
                                "confidence":r.get("confidence",{}),
                                "latency":r.get("latency",0)})
                            save_current_chat(); st.rerun()
                with c2:
                    if st.button("📋 Copy-ready", key=f"copy_{i}"):
                        st.code(content, language=None)

    if st.session_state.edit_index is not None:
        idx = st.session_state.edit_index
        if idx < len(st.session_state.messages):
            old = st.session_state.messages[idx]
            st.divider(); st.markdown("### ✏️ Edit Message")
            new = st.text_area("Your message", value=old.get("content",""), height=120,
                               key="edited_message")
            c1, c2 = st.columns(2)
            with c1: save = st.button("✅ Save & Regenerate", type="primary", use_container_width=True)
            with c2: cancel = st.button("❌ Cancel", use_container_width=True)
            if cancel:
                st.session_state.edit_index = None; st.rerun()
            if save and new.strip():
                st.session_state.messages[idx]["content"] = new.strip()
                st.session_state.messages = st.session_state.messages[:idx+1]
                with st.spinner("Generating..."):
                    try: r = eng.ask_rag(new.strip(), conversation_history=st.session_state.messages)
                    except Exception: r = {"answer":"Failed.","citations":[],"confidence":{}}
                st.session_state.messages.append({"role":"assistant",
                    "content":r.get("answer",""), "citations":r.get("citations",[]),
                    "confidence":r.get("confidence",{}), "latency":r.get("latency",0)})
                st.session_state.edit_index = None; save_current_chat(); st.rerun()

    if st.session_state.messages and st.session_state.messages[-1].get("role") == "assistant":
        st.markdown("**💡 You may also ask**")
        cols = st.columns(2)
        for i, s in enumerate(["Explain this in simple words","Give me the key points",
                               "What are the important concepts?","Give me an example"]):
            with cols[i % 2]:
                if st.button(s, key=f"sug_{i}"):
                    st.session_state.suggested_prompt = s; st.rerun()

    prompt = st.chat_input("Ask anything about your documents...")
    if voice_prompt: prompt = voice_prompt
    if st.session_state.get("suggested_prompt"):
        prompt = st.session_state.pop("suggested_prompt")

    if prompt:
        if not eng.ready:
            st.warning("Please upload and process documents first."); st.stop()
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🔎 Searching your documents..."):
                try: r = eng.ask_rag(prompt, conversation_history=st.session_state.messages)
                except Exception:
                    r = {"answer":"I couldn't generate an answer.","citations":[],
                         "confidence":{"label":"Low","score":0},"latency":0}
            ans = r.get("answer","No answer."); cits = r.get("citations",[])
            conf = r.get("confidence",{})
            render_ai_answer(ans); render_citations(cits); render_confidence(conf)
            st.caption(f"Response time: {r.get('latency',0)} seconds")
        st.session_state.messages.append({"role":"assistant","content":ans,
            "citations":cits, "confidence":conf, "latency":r.get("latency",0)})
        save_current_chat(); log_event("ask_question")

# ---------- COMPARE ----------
elif mode == "Compare":
    st.subheader("🔍 Document Comparison")
    st.caption("Compare information from two uploaded documents.")
    if not eng.ready: st.info("Upload and process documents first.")
    elif len(eng.documents) < 2: st.warning("Comparison requires at least two documents.")
    else:
        da = st.selectbox("Document A", eng.documents)
        db = st.selectbox("Document B", [d for d in eng.documents if d != da])
        q = st.text_input("What would you like to compare?",
                          placeholder="Example: What changed between these documents?")
        if st.button("🔍 Generate Comparison", type="primary", use_container_width=True):
            if not q.strip(): st.warning("Enter a comparison question.")
            else:
                with st.spinner("Analyzing both documents..."):
                    try: r = eng.compare_documents(da, db, q)
                    except Exception as e: st.error(f"Error: {e}"); r = None
                if r and r.get("success"):
                    st.markdown(f"**{da}** ↔ **{db}**"); st.divider()
                    render_ai_answer(r.get("answer",""))
                    render_confidence(r.get("confidence",{}), "Comparison confidence")
                    render_evidence(r.get("evidence",[]), "📚 Comparison Evidence")
                elif r: st.warning(r.get("answer","Unable to compare."))

# ---------- STUDY ----------
elif mode == "Study":
    st.subheader("📖 Study Mode")
    st.caption("Turn your documents into useful study material.")
    stype = st.selectbox("Choose study material",
        ["Multiple Choice Questions","Flashcards","Viva Questions","Revision Notes"])
    topic = st.text_input("Topic", placeholder="Example: Artificial Intelligence")
    if st.button("✨ Generate Study Material", type="primary", use_container_width=True):
        if not eng.ready: st.warning("Please process documents first.")
        elif not topic.strip(): st.warning("Enter a topic first.")
        else:
            with st.spinner("Preparing study material..."):
                try: r = eng.generate_study_material(topic.strip(), stype, 700)
                except Exception as e: st.error(f"Error: {e}"); r = None
            if r:
                render_ai_answer(r.get("content",""))
                render_evidence(r.get("evidence",[]), "📚 Study Sources & Evidence")
                render_confidence(r.get("confidence",{}), "Study grounding confidence")

# ---------- QUIZ ----------
elif mode == "Quiz":
    st.subheader("🎯 Quiz Mode")
    st.caption("Test yourself using questions generated from your uploaded documents.")

    def parse_quiz(content):
        if not content: return []
        content = re.sub(r"```(?:json|text|markdown)?", "", content, flags=re.IGNORECASE)
        content = content.replace("```","").replace("\r","")
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        qs, cur_q, cur_opt, cur_cor = [], None, {}, None
        def save():
            nonlocal cur_q, cur_opt, cur_cor
            if cur_q and len(cur_opt) == 4 and cur_cor in cur_opt:
                qs.append({"question":cur_q.strip(), "options":dict(cur_opt), "correct":cur_cor})
            cur_q, cur_opt, cur_cor = None, {}, None
        for line in lines:
            cl = re.sub(r"^[\*\#\-\•\s]+", "", line)
            cl = re.sub(r"\*\*|__", "", cl).strip()
            m = None
            for pat in [r"^(?:Q(?:uestion)?\s*)?(\d+)\s*[\.\):\-]\s*(.+)$",
                        r"^Question\s+(\d+)\s*[\:\.\-\)]\s*(.+)$",
                        r"^Q\s*(\d+)\s*[\:\.\-\)]\s*(.+)$"]:
                m = re.match(pat, cl, re.IGNORECASE)
                if m: break
            if m: save(); cur_q = m.group(2).strip(); continue
            om = re.match(r"^\(?([A-Da-d])\)?\s*[\.\):\-]\s*(.+)$", cl)
            if om and cur_q:
                cur_opt.setdefault(om.group(1).upper(), om.group(2).strip()); continue
            am = re.match(r"^(?:Correct\s*Answer|Correct\s*Option|Answer|Correct)\s*[\:\-\=]?\s*"
                          r"(?:Option\s*)?[\(\[]?([A-Da-d])[\)\]]?", cl, re.IGNORECASE)
            if am and cur_q: cur_cor = am.group(1).upper(); continue
            if cur_q and not cur_opt and cur_cor is None: cur_q += " " + cl
        save()
        seen, uniq = set(), []
        for q in qs:
            k = q["question"].lower().strip()
            if k not in seen: seen.add(k); uniq.append(q)
        return uniq

    for k, v in [("quiz_questions",[]),("quiz_answers",{}),
                 ("quiz_submitted",False),("quiz_score",0)]:
        if k not in st.session_state: st.session_state[k] = v

    if not eng.ready: st.info("Upload and process documents first.")
    else:
        topic = st.text_input("Quiz topic", placeholder="Example: Machine Learning")
        n = st.slider("Number of questions", 3, 10, 5)

        if st.button("🎯 Generate Quiz", type="primary", use_container_width=True):
            if not topic.strip(): st.warning("Enter a quiz topic.")
            else:
                st.session_state.quiz_questions, st.session_state.quiz_answers = [], {}
                st.session_state.quiz_submitted, st.session_state.quiz_score = False, 0
                quiz_prompt = f"""Create exactly {n} multiple-choice quiz questions.
Topic: {topic.strip()}
Generate ONLY from uploaded documents.
FORMAT:
1. Question text
A) Option one
B) Option two
C) Option three
D) Option four
Correct Answer: B
RULES: exactly {n} questions, four options A-D, one correct, "Correct Answer: X" line,
no explanations, no intro, no summary, no markdown, no bold, no tables, no JSON."""
                with st.spinner("Generating quiz..."):
                    try: r = eng.generate_study_material(topic.strip(), quiz_prompt, 2000)
                    except Exception as e: st.error(f"Quiz error: {e}"); r = None
                if r:
                    parsed = parse_quiz(r.get("content",""))
                    valid = [q for q in parsed
                             if len(q.get("options",{})) == 4
                             and set(q["options"]) == {"A","B","C","D"}
                             and q.get("correct") in ["A","B","C","D"]]
                    if not valid:
                        st.error("Quiz questions could not be read. Try again.")
                    else:
                        if len(valid) < n:
                            st.warning(f"Only {len(valid)} of {n} questions generated. Try again for full set.")
                        st.session_state.quiz_questions = valid[:n]
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_score = 0
                        st.success(f"✅ {len(valid[:n])} questions ready!")

        qs = st.session_state.quiz_questions
        if qs:
            st.divider(); st.markdown("### 📝 Attempt the Quiz")
            st.caption("Select one option per question, then Submit Quiz.")
            for i, qd in enumerate(qs):
                st.markdown(f"### Question {i+1}")
                st.markdown(qd["question"])
                letters = ["A","B","C","D"]
                labels = [qd["options"][L] for L in letters]
                sel = st.radio("Choose:", labels, index=None,
                               key=f"quiz_q_{i}", label_visibility="collapsed")
                if sel is not None:
                    for L in letters:
                        if qd["options"][L] == sel:
                            st.session_state.quiz_answers[i] = L; break
                if st.session_state.quiz_submitted:
                    cor = qd.get("correct"); ua = st.session_state.quiz_answers.get(i)
                    if ua is None: st.warning("⚠️ Not answered")
                    elif ua == cor: st.success(f"✅ Correct! Answer: {cor}")
                    else: st.error(f"❌ Incorrect. Correct answer: {cor}")
                st.divider()

            if not st.session_state.quiz_submitted:
                if st.button("✅ Submit Quiz", type="primary", use_container_width=True):
                    missing = [i+1 for i in range(len(qs)) if i not in st.session_state.quiz_answers]
                    if missing:
                        st.warning(f"Answer all questions. Missing: {', '.join(map(str,missing))}")
                    else:
                        score = sum(1 for i, qd in enumerate(qs)
                                    if st.session_state.quiz_answers.get(i) == qd.get("correct"))
                        st.session_state.quiz_score = score
                        st.session_state.quiz_submitted = True
                        log_event("quiz_completed"); st.rerun()

            if st.session_state.quiz_submitted:
                total, score = len(qs), st.session_state.quiz_score
                pct = score / total * 100
                st.divider(); st.markdown("## 🏆 Quiz Result")
                st.metric("Score", f"{score}/{total}")
                st.progress(pct / 100)
                st.write(f"**Percentage: {pct:.1f}%**")
                if pct >= 80: st.success("🎉 Excellent work!")
                elif pct >= 60: st.info("👍 Good job! Keep practicing.")
                else: st.warning("📚 Keep revising and try again.")
                if st.button("🔄 Try New Quiz", use_container_width=True):
                    st.session_state.quiz_questions, st.session_state.quiz_answers = [], {}
                    st.session_state.quiz_submitted, st.session_state.quiz_score = False, 0
                    st.rerun()

# ---------- ANALYTICS ----------
if st.session_state.get("show_analytics", False):
    st.sidebar.divider(); st.sidebar.subheader("📊 Usage Analytics")
    c1, c2, c3 = st.sidebar.columns(3)
    with c1: st.metric("Chats", len(st.session_state.chat_history))
    with c2: st.metric("Messages", len(st.session_state.messages))
    with c3: st.metric("Docs", len(eng.documents) if eng.ready else 0)
    st.sidebar.caption("Analytics stored locally.")
