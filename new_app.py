"""
Complete Integration: Arabic Legal RAG + Gemini Flash 2.0 + Streamlit
With BOTH Google Drive download AND direct upload options

Installation:
pip install streamlit google-generativeai gdown requests

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
import gdown
import zipfile
import os
import shutil
import requests

# Import your RAG system
try:
    from allin_one import ArabicLegalRAG, DocumentType
except ImportError:
    st.error("❌ Could not import RAG system. Make sure allin_one.py is in the same directory.")
    st.stop()


# ============================================================================
# GOOGLE DRIVE DOWNLOADER
# ============================================================================

class GoogleDriveDownloader:
    """Download and extract legal_index from Google Drive"""

    @staticmethod
    def download_large_file_from_gdrive(file_id: str, destination: str) -> bool:
        """Download large file from Google Drive with proper handling"""

        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params={'id': file_id}, stream=True)
        token = None

        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        # Save with progress
        total_size = int(response.headers.get('content-length', 0))
        block_size = 32768  # 32KB chunks

        progress_bar = st.progress(0)
        status_text = st.empty()

        with open(destination, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = int((downloaded / total_size) * 100)
                        progress_bar.progress(progress / 100)
                        status_text.text(f"Downloaded: {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB")

        progress_bar.empty()
        status_text.empty()
        return True

    @staticmethod
    def download_zip_and_extract(zip_file_id: str, output_dir: str = "legal_index1") -> bool:
        """
        Download a ZIP file from Google Drive and extract it
        Handles large files (200MB+) properly
        """
        try:
            st.info("📥 Starting download from Google Drive...")
            st.warning("⏰ Large file detected - this may take 2-5 minutes. Please wait...")

            # Download ZIP file
            zip_path = "legal_index_temp.zip"

            # Method 1: Try gdown with confirmation bypass
            try:
                st.text("📦 Attempting download (Method 1: gdown)...")
                url = f"https://drive.google.com/uc?id={zip_file_id}"
                gdown.download(url, zip_path, quiet=False, fuzzy=True)

                if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000:
                    raise Exception("Downloaded file is too small or doesn't exist")

            except Exception as e1:
                st.warning(f"Method 1 failed: {str(e1)[:100]}")
                st.text("📦 Trying alternative method (Method 2: requests)...")

                # Method 2: Use requests with virus scan bypass
                downloader = GoogleDriveDownloader()
                success = downloader.download_large_file_from_gdrive(zip_file_id, zip_path)

                if not success:
                    raise Exception("Both download methods failed")

            # Check file size
            file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
            st.success(f"✅ Downloaded {file_size_mb:.1f} MB successfully!")

            # Extract ZIP
            st.info("📂 Extracting files...")
            progress_bar = st.progress(0)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                total_files = len(file_list)
                st.text(f"Found {total_files} files in archive")

                # Extract all files
                for i, file in enumerate(file_list):
                    zip_ref.extract(file, ".")
                    if i % 10 == 0:  # Update progress every 10 files
                        progress_bar.progress((i + 1) / total_files)

                progress_bar.progress(1.0)

            progress_bar.empty()

            # Handle nested folder structure
            if Path("legal_index1/legal_index1").exists():
                st.info("🔧 Fixing folder structure...")
                temp_dir = "legal_index1_temp"
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                shutil.move("legal_index1/legal_index1", temp_dir)
                shutil.rmtree("legal_index1")
                shutil.move(temp_dir, "legal_index1")

            # Clean up ZIP
            if os.path.exists(zip_path):
                os.remove(zip_path)
                st.text("🧹 Cleaned up temporary files")

            # Verify critical files
            st.info("🔍 Verifying extracted files...")

            # Check for different possible metadata file names
            metadata_files = ["chunks_metadata.json", "chunks.json", "metadata.json"]
            faiss_file = "faiss_index.bin"

            # Check FAISS index
            faiss_path = Path(output_dir) / faiss_file
            if not faiss_path.exists():
                st.error(f"❌ Required file missing: {faiss_file}")
                st.info("Files in directory:")
                for item in Path(output_dir).iterdir():
                    st.text(f"  - {item.name}")
                return False
            else:
                size = faiss_path.stat().st_size / (1024 * 1024)
                st.text(f"  ✓ {faiss_file}: {size:.1f} MB")

            # Check for metadata file (any variant)
            metadata_found = False
            for metadata_file in metadata_files:
                metadata_path = Path(output_dir) / metadata_file
                if metadata_path.exists():
                    size = metadata_path.stat().st_size / (1024 * 1024)
                    st.text(f"  ✓ {metadata_file}: {size:.1f} MB")
                    metadata_found = True
                    break

            if not metadata_found:
                st.error(f"❌ No metadata file found. Looking for: {', '.join(metadata_files)}")
                st.info("Files in directory:")
                for item in Path(output_dir).iterdir():
                    st.text(f"  - {item.name}")
                return False

            st.success("✅ All files verified successfully!")
            return True

        except zipfile.BadZipFile:
            st.error("❌ Downloaded file is corrupted or not a valid ZIP")
            st.info("Please check your Google Drive file and try again")
            return False
        except Exception as e:
            st.error(f"❌ Download/Extraction failed: {str(e)}")
            st.exception(e)
            return False


# ============================================================================
# DIRECT FILE UPLOAD HANDLER
# ============================================================================

class DirectUploadHandler:
    """Handle direct ZIP file uploads from user's PC"""

    @staticmethod
    def upload_and_extract(uploaded_file, output_dir: str = "legal_index1") -> bool:
        """Extract uploaded ZIP file"""
        try:
            # Check file size
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"📦 File size: {file_size_mb:.1f} MB")

            if file_size_mb > 500:
                st.error("❌ File too large! Maximum 500MB allowed.")
                return False

            st.info("⏳ Extracting files... This may take a few minutes.")

            # Save uploaded file temporarily
            temp_zip = "temp_legal_index.zip"
            with open(temp_zip, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Remove old directory if exists
            if Path(output_dir).exists():
                st.text("🗑️ Removing old index...")
                shutil.rmtree(output_dir)

            # Extract with progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                total_files = len(file_list)
                status_text.text(f"Extracting {total_files} files...")

                for i, file in enumerate(file_list):
                    zip_ref.extract(file, ".")
                    if i % 10 == 0:
                        progress_bar.progress((i + 1) / total_files)

                progress_bar.progress(1.0)

            progress_bar.empty()
            status_text.empty()

            # Handle nested folders
            if Path("legal_index1/legal_index1").exists():
                st.text("🔧 Fixing folder structure...")
                temp_dir = "legal_index1_temp"
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                shutil.move("legal_index1/legal_index1", temp_dir)
                shutil.rmtree("legal_index1")
                shutil.move(temp_dir, "legal_index1")

            # Clean up
            os.remove(temp_zip)
            st.text("🧹 Cleaned up temporary files")

            # Verify files
            st.info("🔍 Verifying extracted files...")

            # Check for different possible metadata file names
            metadata_files = ["chunks_metadata.json", "chunks.json", "metadata.json"]
            faiss_file = "faiss_index.bin"

            # Check FAISS index
            faiss_path = Path(output_dir) / faiss_file
            if not faiss_path.exists():
                st.error(f"❌ Required file missing: {faiss_file}")
                st.info("Files found in directory:")
                for item in Path(output_dir).iterdir():
                    st.text(f"  - {item.name}")
                return False
            else:
                size = faiss_path.stat().st_size / (1024 * 1024)
                st.text(f"  ✓ {faiss_file}: {size:.1f} MB")

            # Check for metadata file (any variant)
            metadata_found = False
            for metadata_file in metadata_files:
                metadata_path = Path(output_dir) / metadata_file
                if metadata_path.exists():
                    size = metadata_path.stat().st_size / (1024 * 1024)
                    st.text(f"  ✓ {metadata_file}: {size:.1f} MB")
                    metadata_found = True
                    break

            if not metadata_found:
                st.error(f"❌ No metadata file found. Looking for: {', '.join(metadata_files)}")
                st.info("Files found in directory:")
                for item in Path(output_dir).iterdir():
                    st.text(f"  - {item.name}")
                return False

            st.success("✅ Files extracted and verified successfully!")
            return True

        except zipfile.BadZipFile:
            st.error("❌ Invalid ZIP file. Please check your file and try again.")
            return False
        except Exception as e:
            st.error(f"❌ Extraction failed: {str(e)}")
            st.exception(e)
            return False


# ============================================================================
# INDEX SETUP PAGE
# ============================================================================

def render_index_setup_page():
    """Show setup page with both Google Drive and Upload options"""

    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">⚖️ إعداد قاعدة البيانات القانونية</h1>
        <p style="margin: 5px 0; opacity: 0.9;">اختر طريقة تحميل البيانات</p>
    </div>
    """, unsafe_allow_html=True)

    st.warning("⚠️ قاعدة البيانات القانونية غير موجودة. اختر أحد الخيارات التالية:")

    # Create tabs for different methods
    tab1, tab2, tab3 = st.tabs(["☁️ تنزيل من Google Drive", "📤 رفع من الجهاز", "ℹ️ معلومات"])

    # ===== TAB 1: GOOGLE DRIVE =====
    with tab1:
        st.markdown("### ☁️ التنزيل من Google Drive")

        st.info("""
        **الخطوات:**
        1. ضغط مجلد `legal_index1` إلى ملف ZIP
        2. رفع الملف إلى Google Drive
        3. مشاركة الملف (Anyone with the link → Viewer)
        4. نسخ معرف الملف من الرابط
        5. إضافته في إعدادات Streamlit Secrets
        """)

        # Check if secrets configured
        gdrive_id = st.secrets.get("GDRIVE_ZIP_ID", "")

        if gdrive_id and gdrive_id != "YOUR_ZIP_FILE_ID_HERE":
            st.success(f"✅ تم العثور على معرف Google Drive: `{gdrive_id[:20]}...`")

            if st.button("🚀 بدء التنزيل من Google Drive", type="primary", use_container_width=True):
                with st.spinner("جاري التنزيل والاستخراج..."):
                    downloader = GoogleDriveDownloader()
                    success = downloader.download_zip_and_extract(gdrive_id)

                    if success:
                        st.session_state.index_ready = True
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("فشل التنزيل. جرب خيار الرفع المباشر.")
        else:
            st.warning("❌ لم يتم تكوين Google Drive")

            st.code("""
# أضف في Streamlit Cloud → Settings → Secrets:

GDRIVE_ZIP_ID = "1ZhlIWykfRJr65nscaFLWq3dlGaIAym63"
GEMINI_API_KEY = "your_gemini_api_key_here"
            """, language="toml")

            st.markdown("**للحصول على معرف الملف:**")
            st.markdown("من الرابط: `https://drive.google.com/file/d/FILE_ID/view`")
            st.markdown("انسخ الجزء `FILE_ID`")

    # ===== TAB 2: DIRECT UPLOAD =====
    with tab2:
        st.markdown("### 📤 الرفع المباشر من الجهاز")

        st.info("""
        **تعليمات:**
        1. قم بضغط مجلد `legal_index1` إلى ملف ZIP
        2. تأكد من أن الملف يحتوي على:
           - `faiss_index.bin` (مطلوب)
           - `chunks.json` أو `chunks_metadata.json` أو `metadata.json` (مطلوب)
        3. ارفع الملف أدناه (الحد الأقصى: 500 ميجابايت)
        """)

        st.warning("⚠️ **ملاحظة:** الملفات المرفوعة ستحذف عند إعادة تشغيل التطبيق. استخدم Google Drive للاستخدام الدائم.")

        uploaded_file = st.file_uploader(
            "اختر ملف legal_index1.zip",
            type=['zip'],
            help="ملف ZIP يحتوي على قاعدة البيانات القانونية"
        )

        if uploaded_file is not None:
            col1, col2 = st.columns([3, 1])

            with col1:
                st.info(f"📦 الملف: {uploaded_file.name}")
                st.info(f"📊 الحجم: {uploaded_file.size / (1024*1024):.1f} ميجابايت")

            with col2:
                if st.button("⬆️ رفع واستخراج", type="primary", use_container_width=True):
                    handler = DirectUploadHandler()
                    success = handler.upload_and_extract(uploaded_file)

                    if success:
                        st.session_state.index_ready = True
                        st.balloons()
                        time.sleep(2)
                        st.rerun()

    # ===== TAB 3: INFO =====
    with tab3:
        st.markdown("### ℹ️ معلومات حول قاعدة البيانات")

        st.markdown("""
        **الملفات المطلوبة في legal_index1:**
        - `faiss_index.bin` - قاعدة بيانات البحث المتجهي (مطلوب)
        - `chunks.json` أو `chunks_metadata.json` - معلومات النصوص القانونية (مطلوب)
        - ملفات إضافية حسب إعداد RAG الخاص بك
        
        **حجم الملف النموذجي:**
        - 200-250 ميجابايت تقريباً
        
        **الطرق المدعومة:**
        
        1. **Google Drive (موصى به):**
           - ✅ دائم - لا يحذف عند إعادة التشغيل
           - ✅ أسرع للمستخدمين المتعددين
           - ✅ تنزيل تلقائي عند البدء
           - ❌ يتطلب إعداد أولي
        
        2. **الرفع المباشر:**
           - ✅ سهل وسريع
           - ✅ لا يتطلب Google Drive
           - ❌ يحذف عند إعادة تشغيل التطبيق
           - ❌ يجب إعادة الرفع في كل مرة
        
        **التوصية:** استخدم Google Drive للإنتاج، والرفع المباشر للاختبار.
        """)

        with st.expander("🔧 استكشاف الأخطاء"):
            st.markdown("""
            **مشاكل شائعة:**
            
            1. **"Missing required files"**
               - تأكد من وجود `faiss_index.bin` و أحد ملفات البيانات: `chunks.json` أو `chunks_metadata.json`
               - تحقق من بنية المجلد داخل الـ ZIP
            
            2. **"Download failed from Google Drive"**
               - تحقق من أن الملف مشارك بشكل عام (Anyone with link)
               - تأكد من صحة معرف الملف
               - جرب الرفع المباشر كبديل
            
            3. **"File too large"**
               - الحد الأقصى: 500 ميجابايت للرفع المباشر
               - استخدم Google Drive للملفات الأكبر
            
            4. **"Bad ZIP file"**
               - أعد إنشاء ملف ZIP
               - تأكد من عدم تلف الملف أثناء الرفع
            """)


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

        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        self.model = genai.GenerativeModel(
            'gemini-2.0-flash',
            safety_settings=self.safety_settings
        )

        self.translation_model = genai.GenerativeModel(
            'gemini-2.0-flash',
            safety_settings=self.safety_settings
        )

        self.chat = None

    def translate_to_english(self, arabic_text: str, max_tokens: int = 500) -> str:
        """Translate Arabic response to English - Optimized for quota"""
        try:
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

            context.append(f"""
📄 **المصدر {i}** ({result['score']:.0%})
- {result['document_type']} - {result['law_type']}
- {article_info}

{result['text'][:600]}
{'─' * 30}
""")

        full_context = "\n".join(context)

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
proivde english translation too"""





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

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )

            if response.prompt_feedback.block_reason:
                return self._handle_blocked_response(response.prompt_feedback.block_reason), None

            arabic_response = response.text

            english_translation = None
            if include_translation:
                try:
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
        return """## ⚠️ تنبيه الأمان

تم حظر هذا الاستعلام مؤقتاً بواسطة نظام الأمان.

### ✅ الحل:
المصادر القانونية الكاملة متوفرة أدناه مباشرة!

### 💡 نصائح:
- أعد صياغة السؤال بشكل محايد
- استخدم "ما هي الأحكام القانونية..." بدلاً من "ما عقوبة..."
- أو استخدم وضع النصوص المباشرة (بدون Gemini)"""

    def _handle_blocked_response(self, block_reason) -> str:
        return f"""## ⚠️ محظور: {block_reason}

المصادر القانونية الكاملة متوفرة أدناه. اطلع عليها مباشرة."""


# ============================================================================
# RAG SYSTEM LOADER
# ============================================================================

@st.cache_resource
def load_rag_system():
    """Load RAG system"""
    try:
        index_path = Path("legal_index1")

        if not (index_path / "faiss_index.bin").exists():
            return None, False

        with st.spinner("⏳ جاري تحميل قاعدة البيانات القانونية..."):
            rag = ArabicLegalRAG(
                chunk_size=1200,
                overlap=150,
                model_name="intfloat/multilingual-e5-base",
                use_metadata_context=True
            )

            rag.load_knowledge_base("legal_index1")

            st.success(f"✅ تم تحميل {len(rag.embedding_system.chunks)} جزء قانوني")
            return rag, True

    except Exception as e:
        st.error(f"❌ فشل تحميل النظام: {str(e)}")
        st.exception(e)
        return None, False


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_sidebar():
    """Enhanced sidebar"""
    with st.sidebar:
        st.markdown("# ⚙️ **لوحة التحكم**")

        api_key = st.secrets.get("GEMINI_API_KEY", "")

        if api_key:
            st.session_state.api_key = api_key
            st.success("✅ تم حفظ المفتاح")
        else:
            st.info("💡 تحتاج إلى مفتاح API للبدء")

        st.markdown("---")

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

        st.session_state.settings = {
            'threshold': threshold,
            'top_k': top_k,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'filter_type': doc_type_map[filter_type_str],
            'filter_law': filter_law if filter_law else None
        }

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 مسح", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.button("📊 إحصائيات", use_container_width=True):
                st.session_state.show_stats = not st.session_state.get('show_stats', False)

        st.markdown("---")

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

    # Initialize session state
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

    # Check if index exists
    index_path = Path("legal_index1")
    index_exists = (index_path / "faiss_index.bin").exists()

    # If index doesn't exist, show setup page
    if not index_exists and not st.session_state.get('index_ready', False):
        render_index_setup_page()
        return

    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">⚖️ المساعد القانوني الذكي</h1>
        <p style="margin: 5px 0; opacity: 0.9;">مدعوم بتقنية RAG وذكاء Gemini Flash 2.0</p>
    </div>
    """, unsafe_allow_html=True)

    render_sidebar()

    # Load RAG system
    if 'rag_system' not in st.session_state:
        rag, success = load_rag_system()
        if success:
            st.session_state.rag_system = rag
        else:
            st.error("❌ فشل تحميل قاعدة البيانات")
            if st.button("↩️ العودة للإعداد"):
                st.session_state.index_ready = False
                st.rerun()
            st.stop()

    render_stats(st.session_state.rag_system)

    # Chat history display
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "⚖️"):
            if message["role"] == "assistant":
                st.markdown("### 🇸🇦 الإجابة بالعربية")
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("translation"):
                st.markdown("---")
                st.markdown("### 🇬🇧 English Translation")
                st.markdown(message["translation"])

            if message["role"] == "assistant" and "sources" in message:
                if message.get("rag_only"):
                    st.caption("💡 وضع العرض المباشر")
                else:
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
            st.error("⚠️ يرجى إدخال مفتاح Gemini API في الشريط الجانبي أولاً")
            st.stop()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("🔍 جاري البحث في القوانين والمراجع..."):

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
                        st.warning("⚠️ لم يتم العثور على مصادر قانونية ذات صلة")
                        st.stop()

                    use_gemini = st.session_state.get('use_gemini', True)

                    if not use_gemini:
                        # RAG-only mode
                        st.markdown("### 📚 النصوص القانونية ذات الصلة:")

                        for i, result in enumerate(results, 1):
                            article_info = f"المادة {result.get('article', 'غير محدد')}" if result.get('article') else "نص قانوني"

                            with st.expander(f"📄 المصدر {i}: {result['law_type']} - {article_info}", expanded=(i==1)):
                                st.markdown(f"**نوع الوثيقة**: {result['document_type']}")
                                st.markdown(f"**القانون**: {result['law_type']}")
                                st.markdown(format_score(result['score']), unsafe_allow_html=True)
                                st.markdown("---")
                                st.markdown(f"<div style='text-align: right; direction: rtl; line-height: 1.8; background: #f0f2f6; padding: 15px; border-radius: 10px;'>{result['text']}</div>", unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "📚 **تم عرض النصوص القانونية المباشرة**",
                            "sources": results,
                            "rag_only": True
                        })
                        st.stop()

                    # Gemini mode
                    if 'gemini' not in st.session_state:
                        st.session_state.gemini = GeminiLegalAssistant(st.session_state.api_key)

                    gemini_prompt = st.session_state.gemini.create_enhanced_prompt(prompt, results)
                    show_translation = st.session_state.get('show_translation', False)

                    arabic_response, english_translation = st.session_state.gemini.get_response_with_translation(
                        gemini_prompt,
                        temperature=settings['temperature'],
                        max_tokens=settings['max_tokens'],
                        include_translation=show_translation
                    )

                    # Check for errors
                    if "⚠️" in arabic_response or "❌" in arabic_response:
                        st.warning(arabic_response)
                        st.markdown("---")
                        st.markdown("### 📖 النصوص القانونية المباشرة:")

                        for i, result in enumerate(results[:3], 1):
                            with st.expander(f"📄 المصدر {i}", expanded=(i==1)):
                                st.markdown(f"**{result['law_type']}** - المادة {result.get('article', 'غير محدد')}")
                                st.markdown(format_score(result['score']), unsafe_allow_html=True)
                                st.markdown(result['text'][:600])

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": arabic_response,
                            "sources": results
                        })
                        st.stop()

                    # Display response
                    st.markdown("### 🇸🇦 الإجابة بالعربية")
                    st.markdown(arabic_response)

                    if show_translation and english_translation:
                        st.markdown("---")
                        st.markdown("### 🇬🇧 English Translation")
                        st.markdown(english_translation)

                    with st.expander("📚 عرض المصادر القانونية"):
                        render_sources(results)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": arabic_response,
                        "translation": english_translation if show_translation else None,
                        "sources": results
                    })

                except Exception as e:
                    st.error(f"❌ حدث خطأ: {str(e)}")
                    st.exception(e)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>⚖️ النظام القانوني الذكي | مدعوم بـ <strong>Gemini Flash 2.0</strong> و <strong>RAG Technology</strong></p>
        <p style="font-size: 12px;">💡 ملاحظة: هذا النظام للمساعدة فقط وليس بديلاً عن الاستشارة القانونية المهنية</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()