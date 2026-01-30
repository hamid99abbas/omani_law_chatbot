"""
محامي عُمان الذكي - Arabic Legal Chatbot
Streamlit Interface for Omani Legal RAG System
"""

import streamlit as st
import requests
from datetime import datetime
from typing import Optional, Dict, Any
import json

# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = "http://46.62.204.148:8000"

# Arabic RTL CSS
ARABIC_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');

    * {
        font-family: 'Cairo', sans-serif !important;
    }

    /* RTL Support */
    .stApp {
        direction: rtl;
        text-align: right;
    }

    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }

    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 5px 15px;
        margin: 1rem 0;
        margin-left: 20%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .bot-message {
        background: white;
        color: #2d3748;
        padding: 1.5rem;
        border-radius: 15px 15px 15px 5px;
        margin: 1rem 0;
        margin-right: 10%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-right: 4px solid #667eea;
    }

    .error-message {
        background: #fff5f5;
        color: #c53030;
        padding: 1rem;
        border-radius: 10px;
        border-right: 4px solid #fc8181;
        margin: 1rem 0;
    }

    .warning-message {
        background: #fffaf0;
        color: #c05621;
        padding: 1rem;
        border-radius: 10px;
        border-right: 4px solid #f6ad55;
        margin: 1rem 0;
    }

    /* Source cards */
    .source-card {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-right: 3px solid #667eea;
        transition: all 0.3s ease;
    }

    .source-card:hover {
        transform: translateX(-5px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }

    .source-header {
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }

    .source-text {
        color: #4a5568;
        line-height: 1.8;
        margin: 0.5rem 0;
    }

    .source-meta {
        color: #718096;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #e2e8f0;
    }

    /* Stats badges */
    .stat-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.9rem;
    }

    /* Direct answer box */
    .direct-answer {
        background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-right: 5px solid #48bb78;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.2);
    }

    .direct-answer-title {
        color: #48bb78;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }

    .direct-answer-text {
        color: #2d3748;
        font-size: 1.1rem;
        line-height: 1.8;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: white;
        border-left: 2px solid #e2e8f0;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* Input field */
    .stTextInput>div>div>input {
        direction: rtl;
        text-align: right;
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
    }

    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Expander */
    .streamlit-expanderHeader {
        direction: rtl;
        text-align: right;
        background: #f7fafc;
        border-radius: 10px;
        font-weight: 600;
    }

    /* Info box */
    .info-box {
        background: #ebf8ff;
        color: #2c5282;
        padding: 1rem;
        border-radius: 10px;
        border-right: 4px solid #4299e1;
        margin: 1rem 0;
    }

    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
</style>
"""


# ============================================================================
# API Functions
# ============================================================================

def check_api_health() -> Dict[str, Any]:
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"خطأ في الاتصال بالخادم: {str(e)}")
        return None


def query_legal_api(question: str, use_gemini: bool = True) -> Optional[Dict[str, Any]]:
    """Query the legal API"""
    try:
        payload = {
            "question": question,
            "k_laws": 3,
            "k_procedures": 3,
            "k_rulings": 5,
            "use_gemini": use_gemini
        }

        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"خطأ من الخادم: {response.status_code}")
            return None

    except requests.exceptions.Timeout:
        st.error("انتهت مهلة الاتصال. يرجى المحاولة مرة أخرى.")
        return None
    except Exception as e:
        st.error(f"خطأ: {str(e)}")
        return None


# ============================================================================
# UI Components
# ============================================================================

def render_header():
    """Render the header"""
    st.markdown("""
    <div class="header-container">
        <div class="header-title">⚖️ محامي عُمان الذكي</div>
        <div class="header-subtitle">مساعدك القانوني الذكي - نظام استشارات قانونية متقدم</div>
    </div>
    """, unsafe_allow_html=True)


def render_direct_answer(answer: str):
    """Render direct answer"""
    st.markdown(f"""
    <div class="direct-answer">
        <div class="direct-answer-title">📝 الإجابة المباشرة</div>
        <div class="direct-answer-text">{answer}</div>
    </div>
    """, unsafe_allow_html=True)


def render_source_card(source: Dict[str, Any], source_type: str):
    """Render a source card"""
    doc_type_emoji = {
        "law": "📜",
        "procedure": "📋",
        "ruling": "⚖️"
    }

    emoji = doc_type_emoji.get(source_type, "📄")

    # Build header
    header_parts = [source.get('law_type') or source.get('document', 'غير محدد')]

    if source.get('article'):
        header_parts.append(f"المادة {source['article']}")
    if source.get('case_number'):
        header_parts.append(f"الطعن {source['case_number']}")
    if source.get('principle_number'):
        header_parts.append(f"المبدأ {source['principle_number']}")

    header = " - ".join(header_parts)

    # Truncate text
    text = source.get('text', '')
    if len(text) > 400:
        text = text[:400] + "..."

    # Build metadata
    meta_parts = []
    if source.get('year'):
        meta_parts.append(f"السنة: {source['year']}")
    meta_parts.append(f"الدرجة: {source.get('score', 0):.2%}")

    meta = " | ".join(meta_parts)

    st.markdown(f"""
    <div class="source-card">
        <div class="source-header">{emoji} {header}</div>
        <div class="source-text">{text}</div>
        <div class="source-meta">{meta}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sources(sources: Dict[str, Any]):
    """Render all sources"""
    total_sources = sum(len(v) for v in sources.values())

    if total_sources == 0:
        return

    st.markdown("---")
    st.markdown("### 📚 المصادر القانونية")

    # Create tabs for different source types
    tabs = []
    tab_names = []

    if sources.get('laws'):
        tab_names.append(f"📜 القوانين ({len(sources['laws'])})")
        tabs.append(sources['laws'])

    if sources.get('procedures'):
        tab_names.append(f"📋 الإجراءات ({len(sources['procedures'])})")
        tabs.append(sources['procedures'])

    if sources.get('rulings'):
        tab_names.append(f"⚖️ الأحكام ({len(sources['rulings'])})")
        tabs.append(sources['rulings'])

    if tabs:
        tab_objects = st.tabs(tab_names)

        source_types = []
        if sources.get('laws'):
            source_types.append('law')
        if sources.get('procedures'):
            source_types.append('procedure')
        if sources.get('rulings'):
            source_types.append('ruling')

        for tab, source_list, source_type in zip(tab_objects, tabs, source_types):
            with tab:
                for source in source_list:
                    render_source_card(source, source_type)


def render_metadata(metadata: Dict[str, Any]):
    """Render metadata as badges"""
    st.markdown("---")

    badges = []

    if metadata.get('intent'):
        badges.append(f"نوع السؤال: {metadata['intent']}")

    if metadata.get('total_sources'):
        badges.append(f"مجموع المصادر: {metadata['total_sources']}")

    if metadata.get('laws_count'):
        badges.append(f"القوانين: {metadata['laws_count']}")

    if metadata.get('procedures_count'):
        badges.append(f"الإجراءات: {metadata['procedures_count']}")

    if metadata.get('rulings_count'):
        badges.append(f"الأحكام: {metadata['rulings_count']}")

    if metadata.get('gemini_used') is not None:
        badges.append(f"الذكاء الاصطناعي: {'✅' if metadata['gemini_used'] else '❌'}")

    badge_html = " ".join([f'<span class="stat-badge">{badge}</span>' for badge in badges])
    st.markdown(f'<div style="text-align: center; margin: 1rem 0;">{badge_html}</div>', unsafe_allow_html=True)


# ============================================================================
# Main App
# ============================================================================

def main():
    # Page config
    st.set_page_config(
        page_title="محامي عُمان الذكي",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply CSS
    st.markdown(ARABIC_CSS, unsafe_allow_html=True)

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ الإعدادات")

        # API Health Check
        st.markdown("#### 🏥 حالة النظام")
        health = check_api_health()

        if health:
            st.success("✅ النظام يعمل بشكل طبيعي")

            with st.expander("📊 تفاصيل النظام"):
                st.info(f"""
                **إجمالي المستندات:** {health.get('total_chunks', 0):,}

                **أنواع المستندات:**
                - القوانين
                - الإجراءات
                - الأحكام القضائية

                **الذكاء الاصطناعي:** {'✅ مفعّل' if health.get('gemini_configured') else '❌ غير مفعّل'}
                """)
        else:
            st.error("❌ لا يمكن الاتصال بالخادم")

        st.markdown("---")

        # Settings
        use_ai = st.checkbox("استخدام الذكاء الاصطناعي", value=True,
                             help="استخدام Gemini لتوليد إجابات أكثر تفصيلاً")

        st.markdown("---")

        # Clear history
        if st.button("🗑️ مسح المحادثات"):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("---")

        # Examples
        st.markdown("#### 💡 أمثلة للأسئلة")

        example_questions = [
            "ما هي شروط الزواج في القانون العماني؟",
            "كيف يمكنني تقديم شكوى جنائية؟",
            "ما هي عقوبة السرقة؟",
            "المادة 10 من قانون الجزاء العماني",
            "ما هي إجراءات الطلاق؟"
        ]

        for i, q in enumerate(example_questions):
            if st.button(q, key=f"example_{i}"):
                st.session_state.current_question = q

    # Main content
    render_header()

    # Info box
    st.markdown("""
    <div class="info-box">
        <strong>مرحباً بك في محامي عُمان الذكي!</strong><br>
        اطرح سؤالك القانوني وسأقدم لك إجابة مستندة إلى القوانين والأحكام القضائية العمانية.
    </div>
    """, unsafe_allow_html=True)

    # Question input
    col1, col2 = st.columns([5, 1])

    with col1:
        question = st.text_input(
            "اكتب سؤالك القانوني هنا:",
            value=st.session_state.get('current_question', ''),
            placeholder="مثال: ما هي شروط الزواج في القانون العماني؟",
            key="question_input",
            label_visibility="collapsed"
        )

    with col2:
        submit = st.button("إرسال 📤", use_container_width=True)

    # Handle submission
    if submit and question.strip():
        # Clear the example question
        if 'current_question' in st.session_state:
            del st.session_state.current_question

        # Add to history
        st.session_state.chat_history.append({
            "type": "user",
            "content": question,
            "timestamp": datetime.now()
        })

        # Show loading
        with st.spinner("🔍 جاري البحث في القوانين والأحكام..."):
            response = query_legal_api(question, use_gemini=use_ai)

        if response:
            st.session_state.chat_history.append({
                "type": "bot",
                "content": response,
                "timestamp": datetime.now()
            })

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")

        for chat in st.session_state.chat_history:
            if chat["type"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>أنت:</strong><br>
                    {chat['content']}
                </div>
                """, unsafe_allow_html=True)

            else:  # bot
                response = chat["content"]

                # Check if sufficient sources
                if not response.get('has_sufficient_sources'):
                    st.markdown(f"""
                    <div class="warning-message">
                        <strong>⚠️ تنبيه:</strong><br>
                        {response.get('direct_answer', 'عذراً، لا تتوفر معلومات كافية للإجابة على هذا السؤال.')}
                    </div>
                    """, unsafe_allow_html=True)

                    # Show error details if available
                    if response.get('error') == 'article_not_found':
                        error_msg = response.get('error')
                        st.markdown(f"""
                        <div class="error-message">
                            {error_msg}
                        </div>
                        """, unsafe_allow_html=True)

                        if response.get('metadata', {}).get('available_articles'):
                            with st.expander("📋 المواد المتاحة"):
                                articles = response['metadata']['available_articles']
                                st.write(", ".join(map(str, articles)))

                else:
                    # Show direct answer
                    if response.get('direct_answer'):
                        render_direct_answer(response['direct_answer'])

                    # Show full answer in expander
                    if response.get('answer'):
                        with st.expander("📄 عرض الإجابة الكاملة", expanded=False):
                            st.markdown(f"""
                            <div class="bot-message">
                                {response['answer'].replace(chr(10), '<br>')}
                            </div>
                            """, unsafe_allow_html=True)

                    # Show sources
                    if response.get('sources'):
                        render_sources(response['sources'])

                    # Show metadata
                    if response.get('metadata'):
                        render_metadata(response['metadata'])

                st.markdown("<br>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
