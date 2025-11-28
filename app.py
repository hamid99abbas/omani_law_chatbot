"""
Complete Integration: Arabic Legal RAG + Gemini Flash 2.0 + Streamlit
Ready-to-run application - Just add your Gemini API key!

Installation:
pip install streamlit google-generativeai

Run:
streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path
import json
import google.generativeai as genai
from typing import List, Dict, Optional, Tuple
import time

# Import your RAG system
# Make sure your RAG system file is in the same directory or in Python path
try:
    from allin_one import ArabicLegalRAG, DocumentType
except ImportError:
    st.error("❌ Could not import RAG system. Make sure test2.py is in the same directory.")
    st.stop()


# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="⚖️ المساعد القانوني الذكي",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom RTL CSS
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
    
    .legal-source {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .legal-text {
        background: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        backdrop-filter: blur(10px);
    }
    
    .score-badge {
        display: inline-block;
        padding: 8px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        margin: 5px;
    }
    
    .score-excellent { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .score-good { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .score-fair { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 10px;
        font-weight: 600;
    }
    
    h1, h2, h3 {
        color: #667eea;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# GEMINI ASSISTANT
# ============================================================================

class GeminiLegalAssistant:
    """Enhanced Gemini assistant with translation support"""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

        # Safety settings for legal content
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Use Gemini Flash 2.0 (free tier optimized)
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash',
            safety_settings=self.safety_settings
        )

        # Translation model (uses same quota but very efficient)
        self.translation_model = genai.GenerativeModel(
            'gemini-2.0-flash',
            safety_settings=self.safety_settings
        )

        self.chat = None

    def translate_to_english(self, arabic_text: str, max_tokens: int = 500) -> str:
        """Translate Arabic response to English - Optimized for quota"""
        try:
            # Very concise translation prompt to save tokens
            translation_prompt = f"""Translate this Arabic legal text to English concisely:

{arabic_text[:2000]}

Provide ONLY the English translation, no explanations."""

            response = self.translation_model.generate_content(
                translation_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=max_tokens,
                ),
                safety_settings=self.safety_settings
            )

            return response.text

        except Exception as e:
            return f"Translation unavailable: {str(e)[:50]}"

    def create_enhanced_prompt(self, query: str, rag_results: List[Dict]) -> str:
        """Create detailed prompt with RAG context"""

        context = []
        for i, result in enumerate(rag_results, 1):
            article_info = f"المادة {result.get('article', 'غير محدد')}" if result.get('article') else "قرار قضائي"

            # Limit context size to save quota
            context.append(f"""
📄 **المصدر {i}** ({result['score']:.0%})
- {result['document_type']} - {result['law_type']}
- {article_info}

{result['text'][:600]}
{'─' * 30}
""")

        full_context = "\n".join(context)

        # Optimized prompt (shorter to save tokens)
        prompt = f"""أنت مستشار قانوني أكاديمي. قدم إجابة منظمة ودقيقة.

## السؤال:
{query}

## المصادر:
{full_context}

## المطلوب:
1. **الإجابة المباشرة** - خلاصة واضحة
2. **التفصيل القانوني** - اذكر المواد والشروط
3. **التطبيق العملي** - مثال إن أمكن
4. **المصادر المستخدمة**

استخدم تنسيق واضح مع عناوين. كن موجزاً ودقيقاً.

💡 تنويه: معلومات قانونية عامة للإطلاع فقط.
provide english translation no explanation at the end"""

        return prompt

    def get_response_with_translation(self, prompt: str, temperature: float = 0.7,
                                     max_tokens: int = 1500,
                                     include_translation: bool = True) -> Tuple[str, Optional[str]]:
        """Get response with optional English translation"""
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            # Get Arabic response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )

            # Check for blocking
            if response.prompt_feedback.block_reason:
                return self._handle_blocked_response(response.prompt_feedback.block_reason), None

            arabic_response = response.text

            # Get translation if requested
            english_translation = None
            if include_translation:
                try:
                    # Reduce token count for translation
                    english_translation = self.translate_to_english(
                        arabic_response,
                        max_tokens=min(500, max_tokens // 2)
                    )
                except Exception as e:
                    english_translation = f"⚠️ Translation quota exceeded. Try again in a few minutes."

            return arabic_response, english_translation

        except Exception as e:
            error_msg = str(e)

            if 'quota' in error_msg.lower() or 'resource' in error_msg.lower():
                return self._handle_quota_error(), None
            elif 'dangerous_content' in error_msg.lower() or 'safety' in error_msg.lower():
                return self._handle_safety_block(), None
            else:
                return f"❌ خطأ: {error_msg[:100]}", None

    def _handle_quota_error(self) -> str:
        """Handle quota exceeded gracefully"""
        return """## ⏰ تم تجاوز حد الاستخدام المجاني

**الحل السريع**: النصوص القانونية متوفرة أدناه مباشرة! ✅

### 💡 خيارات أخرى:

1. **وضع النصوص المباشرة** (موصى به):
   - اذهب إلى الشريط الجانبي
   - افتح "خيارات متقدمة"  
   - أوقف "استخدام Gemini للإجابات"
   - ستحصل على النصوص القانونية مباشرة

2. **انتظر قليلاً**: 
   - الحصة المجانية: 15 طلب/دقيقة
   - انتظر 2-3 دقائق وحاول مرة أخرى

3. **استخدم مفتاح آخر**:
   - احصل على مفتاح جديد من Google AI Studio
   - مجاني تماماً

📚 **المصادر القانونية متوفرة أدناه بدقة 100%**"""

    def _handle_safety_block(self) -> str:
        """Handle safety blocks"""
        return """## ⚠️ تنبيه الأمان

تم حظر هذا الاستعلام مؤقتاً بواسطة نظام الأمان.

### ✅ الحل:
المصادر القانونية الكاملة متوفرة أدناه مباشرة!

### 💡 نصائح:
- أعد صياغة السؤال بشكل محايد
- استخدم "ما هي الأحكام القانونية..." بدلاً من "ما عقوبة..."
- أو استخدم وضع النصوص المباشرة (بدون Gemini)"""

    def _handle_blocked_response(self, block_reason) -> str:
        """Handle blocked responses"""
        return f"""## ⚠️ محظور: {block_reason}

المصادر القانونية الكاملة متوفرة أدناه. اطلع عليها مباشرة."""


# ============================================================================
# RAG SYSTEM LOADER
# ============================================================================

@st.cache_resource
def load_rag_system():
    """Load RAG system (cached)"""
    try:
        with st.spinner("⏳ جاري تحميل قاعدة البيانات القانونية..."):
            rag = ArabicLegalRAG(
                chunk_size=1200,
                overlap=150,
                model_name="intfloat/multilingual-e5-base",
                use_metadata_context=True
            )

            index_path = Path("legal_index1")
            if (index_path / "faiss_index.bin").exists():
                rag.load_knowledge_base("legal_index1")
                return rag, True
            else:
                return None, False
    except Exception as e:
        st.error(f"❌ فشل تحميل النظام: {str(e)}")
        return None, False


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_sidebar():
    """Enhanced sidebar"""
    with st.sidebar:
        st.markdown("# ⚙️ **لوحة التحكم**")

        # API Key
        api_key = st.secrets["GEMINI_API_KEY"]

        if api_key:
            st.session_state.api_key = api_key
            st.success("✅ تم حفظ المفتاح")
        else:
            st.info("💡 تحتاج إلى مفتاح API للبدء")

        st.markdown("---")

        # Advanced Options
        with st.expander("⚙️ **خيارات متقدمة**"):
            use_gemini = st.checkbox(
                "استخدام Gemini للإجابات",
                value=True,
                help="إذا تم إلغاء التفعيل، سيتم عرض النصوص القانونية مباشرة فقط"
            )

            if use_gemini:
                show_translation = st.checkbox(
                    "🌐 إضافة ترجمة إنجليزية",
                    value=False,
                    help="ترجمة الإجابة إلى الإنجليزية (يستهلك حصة إضافية)"
                )

                show_rag_only = st.checkbox(
                    "عرض النصوص الخام أيضاً",
                    value=False,
                    help="عرض النصوص القانونية الأصلية بالإضافة لإجابة Gemini"
                )
            else:
                show_translation = False
                show_rag_only = True

            st.session_state.use_gemini = use_gemini
            st.session_state.show_translation = show_translation
            st.session_state.show_rag_only = show_rag_only

        st.markdown("---")

        # RAG Settings
        st.markdown("### 🔍 **إعدادات البحث**")

        threshold = st.slider(
            "**حد التطابق**",
            0.0, 1.0, 0.70, 0.05,
            help="كلما زادت القيمة، كانت النتائج أكثر دقة"
        )

        top_k = st.slider(
            "**عدد المصادر**",
            1, 10, 5,
            help="عدد المصادر القانونية المسترجعة"
        )

        st.markdown("---")

        # Gemini Settings
        st.markdown("### 🤖 **إعدادات Gemini**")

        temperature = st.slider(
            "**مستوى الإبداع**",
            0.0, 1.0, 0.7, 0.1,
            help="0 = دقيق، 1 = إبداعي"
        )

        max_tokens = st.slider(
            "**طول الإجابة**",
            500, 4000, 2000, 100,
            help="الحد الأقصى لطول الإجابة"
        )

        st.markdown("---")

        # Filters
        st.markdown("### 🎯 **التصفية**")

        doc_type_map = {
            "الكل": None,
            "قانون": DocumentType.LAW,
            "لائحة تنفيذية": DocumentType.REGULATION,
            "قرار قضائي": DocumentType.JUDICIAL_RULING,
            "غير محدد": DocumentType.UNKNOWN
        }

        filter_type_str = st.selectbox(
            "**نوع الوثيقة**",
            list(doc_type_map.keys())
        )

        filter_law = st.text_input(
            "**اسم القانون** (اختياري)",
            help="مثال: قانون العمل"
        )

        # Save settings
        st.session_state.settings = {
            'threshold': threshold,
            'top_k': top_k,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'filter_type': doc_type_map[filter_type_str],
            'filter_law': filter_law if filter_law else None
        }

        st.markdown("---")

        # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 مسح", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.button("📊 إحصائيات", use_container_width=True):
                st.session_state.show_stats = not st.session_state.get('show_stats', False)

        st.markdown("---")

        # Quick examples
        st.markdown("### 💡 **أمثلة سريعة**")
        examples = [
            "ما شروط فصل العامل؟",
            "اشرح المادة 7 من قانون المقاطعة",
            "ما عقوبة مخالفة قانون العمل؟",
            "حقوق ذوي الإعاقة في التوظيف"
        ]

        for example in examples:
            if st.button(f"📝 {example}", key=example, use_container_width=True):
                st.session_state.example_query = example


def format_score(score: float) -> str:
    """Format score with emoji and color"""
    if score >= 0.85:
        return f'<span class="score-badge score-excellent">🌟 ممتاز {score:.0%}</span>'
    elif score >= 0.70:
        return f'<span class="score-badge score-good">✅ جيد {score:.0%}</span>'
    else:
        return f'<span class="score-badge score-fair">📊 مقبول {score:.0%}</span>'


def render_sources(results: List[Dict]):
    """Render legal sources beautifully"""
    st.markdown("### 📚 المصادر القانونية المستخدمة")

    for i, result in enumerate(results, 1):
        article = result.get('article', 'غير محدد')
        case = result.get('case_number', '')

        ref = f"المادة {article}" if article != 'غير محدد' else f"القضية {case}" if case else "نص عام"

        st.markdown(f"""
        <div class="legal-source">
            <h3 style="color: white; margin: 0;">📄 المصدر {i}</h3>
            <p style="margin: 5px 0;"><strong>النوع:</strong> {result['document_type']}</p>
            <p style="margin: 5px 0;"><strong>القانون:</strong> {result['law_type']}</p>
            <p style="margin: 5px 0;"><strong>المرجع:</strong> {ref}</p>
            <p style="margin: 5px 0;"><strong>المطابقة:</strong> {format_score(result['score'])}</p>
            <div class="legal-text">
                <p style="text-align: right; direction: rtl; line-height: 1.8; margin: 0;">
                    {result['text'][:500]}{'...' if len(result['text']) > 500 else ''}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_stats(rag_system):
    """Render system statistics"""
    if st.session_state.get('show_stats', False):
        st.markdown("### 📊 إحصائيات النظام")

        chunks = rag_system.embedding_system.chunks

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "الوثائق",
                len(set(c.document_name for c in chunks)),
                help="عدد الملفات القانونية"
            )

        with col2:
            st.metric(
                "الأجزاء",
                len(chunks),
                help="إجمالي الأجزاء المفهرسة"
            )

        with col3:
            st.metric(
                "المواد",
                sum(1 for c in chunks if c.article_number),
                help="عدد المواد القانونية"
            )

        with col4:
            st.metric(
                "الأحكام",
                sum(1 for c in chunks if c.case_number),
                help="عدد الأحكام القضائية"
            )

        # Document types
        doc_types = {}
        for chunk in chunks:
            dt = chunk.document_type.value
            doc_types[dt] = doc_types.get(dt, 0) + 1

        st.markdown("#### توزيع أنواع الوثائق")
        for dt, count in sorted(doc_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(chunks)) * 100
            st.progress(percentage / 100, text=f"**{dt}**: {count} ({percentage:.1f}%)")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application"""

    # Title
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">⚖️ المساعد القانوني الذكي</h1>
        <p style="margin: 5px 0; opacity: 0.9;">مدعوم بتقنية RAG وذكاء Gemini Flash 2.0</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'threshold': 0.70,
            'top_k': 5,
            'temperature': 0.7,
            'max_tokens': 2000,
            'filter_type': None,
            'filter_law': None
        }

    # Sidebar
    render_sidebar()

    # Load RAG system
    if 'rag_system' not in st.session_state:
        rag, success = load_rag_system()
        if success:
            st.session_state.rag_system = rag
            st.success("✅ تم تحميل قاعدة البيانات القانونية بنجاح!")
        else:
            st.error("❌ فشل تحميل قاعدة البيانات. تأكد من وجود مجلد legal_index")
            st.stop()

    # Show stats
    render_stats(st.session_state.rag_system)

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "⚖️"):
            # Show Arabic content
            if message["role"] == "assistant":
                st.markdown("### 🇸🇦 الإجابة بالعربية")
            st.markdown(message["content"])

            # Show English translation if available
            if message["role"] == "assistant" and message.get("translation"):
                st.markdown("---")
                st.markdown("### 🇬🇧 English Translation")
                st.markdown(message["translation"])

            if message["role"] == "assistant" and "sources" in message:
                # Check if it's RAG-only mode
                if message.get("rag_only"):
                    st.caption("💡 وضع العرض المباشر - انظر النصوص أعلاه")
                else:
                    # Normal mode with expandable sources
                    with st.expander("📚 عرض المصادر القانونية"):
                        render_sources(message["sources"])

    # Handle example query
    if 'example_query' in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query
        st.rerun()

    # Chat input
    if prompt := st.chat_input("💬 اكتب سؤالك القانوني هنا..."):

        # Check API key
        if 'api_key' not in st.session_state or not st.session_state.api_key:
            st.error("⚠️ **يرجى إدخال مفتاح Gemini API في الشريط الجانبي أولاً**")
            st.stop()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("🔍 جاري البحث في القوانين والمراجع..."):

                # Query RAG
                try:
                    settings = st.session_state.settings
                    results = st.session_state.rag_system.query(
                        prompt,
                        k=settings['top_k'],
                        filter_doc_type=settings['filter_type'],
                        filter_law_type=settings['filter_law'],
                        min_score=settings['threshold']
                    )

                    if not results:
                        st.warning("⚠️ **لم يتم العثور على مصادر قانونية ذات صلة**\n\n💡 جرب:\n- خفض حد التطابق\n- تعديل صياغة السؤال\n- إزالة التصفية")
                        st.stop()

                    # Check if user wants RAG-only mode
                    use_gemini = st.session_state.get('use_gemini', True)
                    show_rag_only = st.session_state.get('show_rag_only', False)

                    if not use_gemini:
                        # RAG-only mode: Show direct results
                        st.markdown("### 📚 النصوص القانونية ذات الصلة:")

                        for i, result in enumerate(results, 1):
                            article_info = f"المادة {result.get('article', 'غير محدد')}" if result.get('article') else "نص قانوني عام"

                            with st.expander(f"📄 المصدر {i}: {result['law_type']} - {article_info}", expanded=(i==1)):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"**نوع الوثيقة**: {result['document_type']}")
                                    st.markdown(f"**القانون**: {result['law_type']}")
                                with col2:
                                    st.markdown(format_score(result['score']), unsafe_allow_html=True)

                                st.markdown("---")
                                st.markdown(f"<div style='text-align: right; direction: rtl; line-height: 1.8; background: #f0f2f6; padding: 15px; border-radius: 10px;'>{result['text']}</div>", unsafe_allow_html=True)

                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "📚 **تم عرض النصوص القانونية المباشرة**",
                            "sources": results,
                            "rag_only": True
                        })

                        st.stop()

                    # Normal Gemini mode
                    # Initialize Gemini
                    if 'gemini' not in st.session_state:
                        st.session_state.gemini = GeminiLegalAssistant(st.session_state.api_key)

                    # Generate response with translation
                    gemini_prompt = st.session_state.gemini.create_enhanced_prompt(prompt, results)

                    # Check if translation is enabled
                    show_translation = st.session_state.get('show_translation', False)

                    arabic_response, english_translation = st.session_state.gemini.get_response_with_translation(
                        gemini_prompt,
                        temperature=settings['temperature'],
                        max_tokens=settings['max_tokens'],
                        include_translation=show_translation
                    )

                    # Check if response indicates failure
                    if "⚠️" in arabic_response or "❌" in arabic_response:
                        if ("محظور" in arabic_response or "خطأ" in arabic_response or
                            "تجاوز" in arabic_response or "استنفدت" in arabic_response):
                            st.warning(arabic_response)

                            # Show direct RAG results as fallback
                            st.markdown("---")
                            st.markdown("### 📖 النصوص القانونية المباشرة:")

                            for i, result in enumerate(results, 1):
                                with st.expander(f"📄 المصدر {i}: {result['law_type']} - المادة {result.get('article', 'غير محدد')}", expanded=(i==1)):
                                    st.markdown(f"**نوع الوثيقة**: {result['document_type']}")
                                    st.markdown(f"**القانون**: {result['law_type']}")
                                    st.markdown(f"**درجة المطابقة**: {format_score(result['score'])}", unsafe_allow_html=True)
                                    st.markdown("---")
                                    st.markdown(f"<div style='text-align: right; direction: rtl; line-height: 1.8;'>{result['text']}</div>", unsafe_allow_html=True)

                            # Save to history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": arabic_response,
                                "sources": results,
                                "fallback": True
                            })
                            st.stop()

                    # Normal successful response
                    # Display Arabic response
                    st.markdown("### 🇸🇦 الإجابة بالعربية")
                    st.markdown(arabic_response)

                    # Display English translation if available
                    if show_translation and english_translation:
                        st.markdown("---")
                        st.markdown("### 🇬🇧 English Translation")
                        if "unavailable" in english_translation.lower() or "exceeded" in english_translation.lower():
                            st.info(english_translation)
                        else:
                            st.markdown(english_translation)

                    # Show sources
                    with st.expander("📚 عرض المصادر القانونية"):
                        render_sources(results)

                    # Save
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": arabic_response,
                        "translation": english_translation if show_translation else None,
                        "sources": results
                    })

                except Exception as e:
                    st.error(f"❌ **حدث خطأ**: {str(e)}")

                    # Show RAG results as ultimate fallback
                    if 'results' in locals() and results:
                        st.info("💡 **عرض المصادر القانونية المباشرة**:")
                        render_sources(results)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>⚖️ النظام القانوني الذكي | مدعوم بـ <strong>Gemini Flash 2.0</strong> و <strong>RAG Technology</strong></p>
        <p style="font-size: 12px;">💡 ملاحظة: هذا النظام للمساعدة فقط وليس بديلاً عن الاستشارة القانونية المهنية</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()