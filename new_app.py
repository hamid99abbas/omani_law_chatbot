"""
Complete Streamlit Frontend for Arabic Legal Assistant
With Firebase Authentication and API Integration

Installation:
pip install streamlit requests python-firebase

Run:
streamlit run app.py

Test User:
Email: hamidatabbas@gmail.com
Password: 92528240
"""

import streamlit as st
import requests
import json
from typing import Optional, List, Dict
from datetime import datetime
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="⚖️ المساعد القانوني الذكي",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = st.secrets.get("API_BASE_URL", "")

# Firebase Web API Key
FIREBASE_WEB_API_KEY = st.secrets.get("FIREBASE_WEB_API_KEY", "")

# ============================================================================
# CUSTOM CSS - RTL SUPPORT
# ============================================================================

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');

    * {
        font-family: 'Cairo', sans-serif;
    }

    .main {
        direction: rtl;
        text-align: right;
    }

    .stTextInput > div > div > input {
        text-align: right;
        direction: rtl;
    }

    .stTextArea > div > div > textarea {
        text-align: right;
        direction: rtl;
    }

    /* Chat message styles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px 20px;
        border-radius: 15px;
        margin: 10px 0;
        color: white;
        text-align: right;
        direction: rtl;
    }

    .assistant-message {
        background: #f0f2f6;
        padding: 15px 20px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: right;
        direction: rtl;
        border-right: 4px solid #667eea;
    }

    .source-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #e0e0e0;
        text-align: right;
        direction: rtl;
    }

    .score-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 12px;
        margin: 5px;
    }

    .score-excellent { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .score-good { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .score-fair { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }

    .header-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
    }

    .stButton > button {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }

    h1, h2, h3 {
        color: #667eea;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# FIREBASE AUTHENTICATION
# ============================================================================

class FirebaseAuth:
    """Handle Firebase authentication"""

    @staticmethod
    def sign_in(email: str, password: str) -> Optional[Dict]:
        """Sign in with email and password"""
        auth_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"

        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }

        try:
            response = requests.post(auth_url, json=payload)
            if response.status_code == 200:
                data = response.json()
                return {
                    "idToken": data["idToken"],
                    "email": data["email"],
                    "localId": data["localId"],
                    "expiresIn": data["expiresIn"]
                }
            else:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "Unknown error")
                st.error(f"❌ فشل تسجيل الدخول: {error_message}")
                return None
        except Exception as e:
            st.error(f"❌ خطأ في الاتصال: {e}")
            return None


# ============================================================================
# API INTERFACE
# ============================================================================

class LegalAssistantAPI:
    """Interface to Legal Assistant API"""

    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

    def query(self, question: str, k: int = 5) -> Optional[Dict]:
        """Send query to API"""
        url = f"{API_BASE_URL}/api/query"

        payload = {
            "query": question,
            "k": k,
            "use_gemini": True,
            "include_translation": False
        }

        try:
            with st.spinner("🔍 جاري البحث في القوانين..."):
                response = requests.post(url, headers=self.headers, json=payload, timeout=30)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 402:
                st.error("⚠️ رصيدك من الاستفسارات انتهى. يرجى ترقية الاشتراك.")
                return None
            else:
                st.error(f"❌ خطأ: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            st.error("⏰ انتهت مهلة الطلب. حاول مرة أخرى.")
            return None
        except Exception as e:
            st.error(f"❌ خطأ: {e}")
            return None

    def get_user_profile(self) -> Optional[Dict]:
        """Get user profile"""
        url = f"{API_BASE_URL}/api/user/profile"

        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None

    def get_conversations(self) -> List[Dict]:
        """Get conversation history"""
        url = f"{API_BASE_URL}/api/conversations"

        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                return data.get("conversations", [])
            return []
        except:
            return []


# ============================================================================
# UI COMPONENTS
# ============================================================================

def format_score(score: float) -> str:
    """Format score with color"""
    if score >= 0.85:
        return f'<span class="score-badge score-excellent">🌟 ممتاز {score:.0%}</span>'
    elif score >= 0.70:
        return f'<span class="score-badge score-good">✅ جيد {score:.0%}</span>'
    else:
        return f'<span class="score-badge score-fair">📊 مقبول {score:.0%}</span>'


def render_source(source: Dict, index: int):
    """Render a single source"""
    article = source.get('article', 'غير محدد')
    ref = f"المادة {article}" if article != 'غير محدد' else "نص عام"

    st.markdown(f"""
    <div class="source-card">
        <h4 style="margin: 0; color: #667eea;">📄 المصدر {index}</h4>
        <p style="margin: 5px 0;"><strong>النوع:</strong> {source['document_type']}</p>
        <p style="margin: 5px 0;"><strong>القانون:</strong> {source['law_type']}</p>
        <p style="margin: 5px 0;"><strong>المرجع:</strong> {ref}</p>
        <p style="margin: 5px 0;"><strong>المطابقة:</strong> {format_score(source['score'])}</p>
        <hr style="margin: 10px 0;">
        <p style="line-height: 1.8; color: #333;">
            {source['text'][:400]}{'...' if len(source['text']) > 400 else ''}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(api: LegalAssistantAPI):
    """Render sidebar with user info"""
    with st.sidebar:
        st.markdown("# ⚙️ **لوحة التحكم**")

        # User profile
        profile = api.get_user_profile()
        if profile and profile.get('success'):
            user_data = profile.get('user', {})
            subscription = user_data.get('subscription', {})

            st.markdown("### 👤 **معلومات الحساب**")
            st.info(f"""
            **📧 البريد:** {st.session_state.user_email}
            **📦 الخطة:** {subscription.get('plan', 'غير محدد')}
            **🎟️ الرصيد:** {subscription.get('tokensRemaining', 0)} استفسار
            **📊 المستخدم:** {subscription.get('tokensUsed', 0)} استفسار
            """)

        st.markdown("---")

        # Quick examples
        st.markdown("### 💡 **أمثلة سريعة**")
        examples = [
            "ما هي العقوبات في قانون مقاطعة إسرائيل؟",
            "ما شروط فصل العامل؟",
            "حقوق ذوي الإعاقة في التوظيف",
            "ما نص المادة السابعة من قانون المقاطعة؟"
        ]

        for example in examples:
            if st.button(f"📝 {example[:35]}...", key=example, use_container_width=True):
                st.session_state.example_query = example
                st.rerun()

        st.markdown("---")

        # Conversation history
        st.markdown("### 📚 **المحادثات السابقة**")
        conversations = api.get_conversations()

        if conversations:
            for conv in conversations[:5]:
                st.text(f"💬 {conv.get('title', 'محادثة')[:30]}...")
        else:
            st.text("لا توجد محادثات سابقة")

        st.markdown("---")

        # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 مسح", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.button("🚪 خروج", use_container_width=True):
                st.session_state.clear()
                st.rerun()


# ============================================================================
# LOGIN PAGE
# ============================================================================

def render_login_page():
    """Render login page"""

    st.markdown("""
    <div class="header-gradient">
        <h1 style="color: white; margin: 0;">⚖️ المساعد القانوني الذكي</h1>
        <p style="margin: 10px 0; opacity: 0.9;">نظام ذكي للاستشارات القانونية</p>
    </div>
    """, unsafe_allow_html=True)

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 🔐 تسجيل الدخول")

        # Auto-fill test credentials
        email = st.text_input(
            "البريد الإلكتروني",
            value="hamidatabbas@gmail.com",
            key="login_email"
        )

        password = st.text_input(
            "كلمة المرور",
            type="password",
            value="92528240",
            key="login_password"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 دخول", use_container_width=True, type="primary"):
            with st.spinner("جاري تسجيل الدخول..."):
                auth_data = FirebaseAuth.sign_in(email, password)

                if auth_data:
                    st.session_state.authenticated = True
                    st.session_state.token = auth_data["idToken"]
                    st.session_state.user_email = auth_data["email"]
                    st.session_state.user_id = auth_data["localId"]
                    st.session_state.messages = []
                    st.success("✅ تم تسجيل الدخول بنجاح!")
                    time.sleep(1)
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("ℹ️ معلومات التطبيق"):
            st.markdown("""
            **المميزات:**
            - 🔍 بحث ذكي في 84,000+ نص قانوني
            - 🤖 إجابات مدعومة بالذكاء الاصطناعي
            - 📚 مصادر قانونية موثوقة
            - 💬 حفظ تاريخ المحادثات

            **للاختبار:**
            - البريد: hamidatabbas@gmail.com
            - كلمة المرور: 92528240
            """)


# ============================================================================
# CHAT PAGE
# ============================================================================

def render_chat_page():
    """Render main chat interface"""

    # Initialize API
    api = LegalAssistantAPI(st.session_state.token)

    # Header
    st.markdown("""
    <div class="header-gradient">
        <h1 style="color: white; margin: 0;">⚖️ المساعد القانوني الذكي</h1>
        <p style="margin: 10px 0; opacity: 0.9;">اسأل أي سؤال قانوني وسأجيبك بدقة</p>
    </div>
    """, unsafe_allow_html=True)

    # Render sidebar
    render_sidebar(api)

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>👤 أنت:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <strong>⚖️ المساعد القانوني:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)

            # Show sources
            if "sources" in message and message["sources"]:
                with st.expander("📚 عرض المصادر القانونية"):
                    for i, source in enumerate(message["sources"], 1):
                        render_source(source, i)

    # Handle example query
    if 'example_query' in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query

        # Add to messages
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        # Get response
        response = api.query(query)

        if response and response.get('success'):
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get('answer', 'لم أتمكن من الحصول على إجابة'),
                "sources": response.get('sources', [])
            })

        st.rerun()

    # Chat input
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])

        with col1:
            user_input = st.text_area(
                "اكتب سؤالك هنا...",
                height=100,
                placeholder="مثال: ما هي العقوبات في قانون مقاطعة إسرائيل؟",
                label_visibility="collapsed"
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("📤 إرسال", use_container_width=True, type="primary")

    if submit and user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Get response from API
        response = api.query(user_input)

        if response and response.get('success'):
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get('answer', 'لم أتمكن من الحصول على إجابة'),
                "sources": response.get('sources', []),
                "tokens_used": response.get('tokens_used', 0),
                "tokens_remaining": response.get('tokens_remaining', 0)
            })

            st.success(f"✅ تم الرد! الرصيد المتبقي: {response.get('tokens_remaining', 0)} استفسار")
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "❌ عذراً، حدث خطأ في معالجة سؤالك. يرجى المحاولة مرة أخرى.",
                "sources": []
            })

        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>⚖️ النظام القانوني الذكي | مدعوم بـ <strong>Gemini Flash 2.0</strong> و <strong>RAG Technology</strong></p>
        <p style="font-size: 12px;">💡 ملاحظة: هذا النظام للمساعدة فقط وليس بديلاً عن الاستشارة القانونية المهنية</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application"""

    # Check authentication
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        render_login_page()
    else:
        render_chat_page()


if __name__ == "__main__":
    main()