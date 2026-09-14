"""
لعبة تعلم كتابة الحروف العربية للأطفال
========================================
تطبيق تفاعلي يعرض على الطفل كلمة بها حرف/حروف ناقصة، فيقوم برسمها على
لوحة رسم، ويتحقق النظام من صحة الحرف باستخدام نموذج تصنيف مدرَّب مسبقًا.

المستويات:
    - المستوى 1: كلمة من 3 حروف، حرف واحد ناقص.
    - المستوى 2: كلمة من 4 حروف، حرفان ناقصان.
    - المستوى 3: كلمة من 5-6 حروف، ثلاثة حروف ناقصة.

الانتقال بين المستويات يتم تلقائيًا بعد إجابة عدد محدد من الكلمات بشكل صحيح.
"""

import random

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ---------------------------------------------------------------------------
# إعدادات عامة وثوابت
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="تعلم كتابة الحروف العربية",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "arabic_letters_model.keras"
CANVAS_SIZE = 200
WORDS_TO_ADVANCE = 3   # عدد الإجابات الصحيحة المطلوبة للانتقال للمستوى التالي
MAX_LEVEL = 3

LETTERS = [
    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز',
    'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك',
    'ل', 'م', 'ن', 'ه', 'و', 'ي',
]

WORD_BANK = {
    1: ["قلم", "باب", "شمس", "قمر", "ولد", "جمل", "نمر",
        "بحر", "نهر", "ليل", "دلو", "درج", "كوب", "قرد", "حمل"],
    2: ["كتاب", "حصان", "تفاح", "دفتر", "قطار", "سرير",
        "غزال", "جمال", "هلال", "بطيخ", "ثعلب", "جدول"],
    3: ["تمساح", "برتقال", "دولاب", "تلفاز"],
}

LEVEL_LABELS = {1: "المستوى الأول", 2: "المستوى الثاني", 3: "المستوى الثالث"}

# ---------------------------------------------------------------------------
# صور توضيحية (SVG مرسومة بالكود لكل كلمة - بدون إيموجي وبدون اعتماد على الإنترنت)
# ---------------------------------------------------------------------------

WORD_ICONS = {
    "قلم": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <line x1="28" y1="92" x2="80" y2="40" stroke="#123B4F" stroke-width="7" stroke-linecap="round"/>
        <polygon points="80,40 96,24 104,32 88,48" fill="#D9A441" stroke="#123B4F" stroke-width="4" stroke-linejoin="round"/>
        <line x1="22" y1="98" x2="30" y2="90" stroke="#123B4F" stroke-width="7" stroke-linecap="round"/>
    </svg>""",

    "باب": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <rect x="34" y="14" width="52" height="94" rx="6" fill="none" stroke="#123B4F" stroke-width="7"/>
        <circle cx="72" cy="62" r="4.5" fill="#D9A441"/>
    </svg>""",

    "شمس": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <circle cx="60" cy="60" r="22" fill="#D9A441"/>
        <g stroke="#D9A441" stroke-width="7" stroke-linecap="round">
            <line x1="60" y1="8" x2="60" y2="24"/><line x1="60" y1="96" x2="60" y2="112"/>
            <line x1="8" y1="60" x2="24" y2="60"/><line x1="96" y1="60" x2="112" y2="60"/>
            <line x1="24" y1="24" x2="35" y2="35"/><line x1="85" y1="85" x2="96" y2="96"/>
            <line x1="96" y1="24" x2="85" y2="35"/><line x1="35" y1="85" x2="24" y2="96"/>
        </g>
    </svg>""",

    "قمر": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <path d="M78 18 A42 42 0 1 0 78 102 A33 33 0 1 1 78 18 Z" fill="#D9A441"/>
    </svg>""",

    "ولد": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <circle cx="60" cy="34" r="17" fill="none" stroke="#123B4F" stroke-width="7"/>
        <path d="M34 104 C34 74 48 58 60 58 C72 58 86 74 86 104" fill="none" stroke="#123B4F" stroke-width="7" stroke-linecap="round"/>
    </svg>""",

    "جمل": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 98 Q14 70 28 66 Q34 48 44 58 Q50 42 60 56 Q68 40 78 58 Q92 62 98 90 L98 100 L12 100 Z"
              fill="none" stroke="#123B4F" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>
        <line x1="28" y1="100" x2="28" y2="110" stroke="#123B4F" stroke-width="6" stroke-linecap="round"/>
        <line x1="84" y1="100" x2="84" y2="110" stroke="#123B4F" stroke-width="6" stroke-linecap="round"/>
    </svg>""",

    "نمر": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <circle cx="60" cy="60" r="30" fill="none" stroke="#123B4F" stroke-width="6"/>
        <circle cx="34" cy="34" r="10" fill="none" stroke="#123B4F" stroke-width="6"/>
        <circle cx="86" cy="34" r="10" fill="none" stroke="#123B4F" stroke-width="6"/>
        <path d="M40 55 Q60 65 80 55 M45 75 L52 82 M75 75 L68 82" fill="none" stroke="#123B4F" stroke-width="5" stroke-linecap="round"/>
    </svg>""",

    "بحر": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <g fill="none" stroke="#123B4F" stroke-width="6" stroke-linecap="round">
            <path d="M10 50 Q30 38 50 50 T90 50 T110 50"/>
            <path d="M10 70 Q30 58 50 70 T90 70 T110 70"/>
            <path d="M10 90 Q30 78 50 90 T90 90 T110 90"/>
        </g>
    </svg>""",

    "نهر": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <line x1="14" y1="20" x2="14" y2="100" stroke="#123B4F" stroke-width="6" stroke-linecap="round"/>
        <line x1="106" y1="20" x2="106" y2="100" stroke="#123B4F" stroke-width="6" stroke-linecap="round"/>
        <path d="M20 45 Q40 35 60 45 T100 45" fill="none" stroke="#D9A441" stroke-width="6" stroke-linecap="round"/>
        <path d="M20 65 Q40 55 60 65 T100 65" fill="none" stroke="#D9A441" stroke-width="6" stroke-linecap="round"/>
        <path d="M20 85 Q40 75 60 85 T100 85" fill="none" stroke="#D9A441" stroke-width="6" stroke-linecap="round"/>
    </svg>""",

    "ليل": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <path d="M70 14 A34 34 0 1 0 70 86 A26 26 0 1 1 70 14 Z" fill="#123B4F"/>
        <g fill="#D9A441"><circle cx="90" cy="95" r="3"/><circle cx="100" cy="80" r="2.2"/><circle cx="80" cy="105" r="2.2"/></g>
    </svg>""",

    "دلو": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <path d="M32 40 L88 40 L78 102 L42 102 Z" fill="none" stroke="#123B4F" stroke-width="6" stroke-linejoin="round"/>
        <path d="M38 40 Q60 10 82 40" fill="none" stroke="#123B4F" stroke-width="6"/>
    </svg>""",

    "درج": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <path d="M14 106 L14 86 L44 86 L44 66 L74 66 L74 46 L106 46 L106 106 Z" fill="none" stroke="#123B4F" stroke-width="6" stroke-linejoin="round"/>
    </svg>""",

    "كوب": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <path d="M30 30 L84 30 L78 96 L36 96 Z" fill="none" stroke="#123B4F" stroke-width="6" stroke-linejoin="round"/>
        <path d="M84 40 Q104 40 104 58 Q104 76 84 76" fill="none" stroke="#123B4F" stroke-width="6"/>
    </svg>""",

    "قرد": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <circle cx="60" cy="64" r="28" fill="none" stroke="#123B4F" stroke-width="6"/>
        <circle cx="30" cy="46" r="12" fill="none" stroke="#123B4F" stroke-width="6"/>
        <circle cx="90" cy="46" r="12" fill="none" stroke="#123B4F" stroke-width="6"/>
        <circle cx="50" cy="60" r="3.5" fill="#123B4F"/><circle cx="70" cy="60" r="3.5" fill="#123B4F"/>
        <path d="M48 78 Q60 86 72 78" fill="none" stroke="#123B4F" stroke-width="5" stroke-linecap="round"/>
    </svg>""",

    "حمل": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <g fill="none" stroke="#123B4F" stroke-width="6">
            <circle cx="45" cy="55" r="14"/><circle cx="62" cy="48" r="16"/><circle cx="80" cy="58" r="14"/>
            <circle cx="55" cy="68" r="15"/><circle cx="72" cy="70" r="14"/>
        </g>
        <circle cx="30" cy="60" r="9" fill="#123B4F"/>
    </svg>""",

    "كتاب": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <path d="M60 26 C50 18 30 18 18 24 L18 92 C30 86 50 86 60 94 C70 86 90 86 102 92 L102 24 C90 18 70 18 60 26 Z"
              fill="none" stroke="#123B4F" stroke-width="6" stroke-linejoin="round"/>
        <line x1="60" y1="26" x2="60" y2="94" stroke="#123B4F" stroke-width="6"/>
    </svg>""",

    "حصان": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <path d="M40 100 L40 66 Q34 50 44 34 Q50 20 66 20 Q78 20 80 32 L92 30 L84 44 Q90 52 88 64 L88 100"
              fill="none" stroke="#123B4F" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>
        <circle cx="70" cy="34" r="3" fill="#123B4F"/>
    </svg>""",

    "تفاح": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <path d="M60 40 C40 24 16 40 18 62 C20 88 42 104 60 104 C78 104 100 88 102 62 C104 40 80 24 60 40 Z" fill="#D9A441"/>
        <path d="M60 40 C60 30 56 22 60 14" fill="none" stroke="#123B4F" stroke-width="5" stroke-linecap="round"/>
        <path d="M60 22 Q72 16 78 26" fill="none" stroke="#123B4F" stroke-width="5" stroke-linecap="round"/>
    </svg>""",

    "دفتر": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <rect x="28" y="16" width="72" height="90" rx="4" fill="none" stroke="#123B4F" stroke-width="6"/>
        <g stroke="#D9A441" stroke-width="5"><circle cx="28" cy="30" r="4"/><circle cx="28" cy="50" r="4"/>
            <circle cx="28" cy="70" r="4"/><circle cx="28" cy="90" r="4"/></g>
        <line x1="44" y1="40" x2="88" y2="40" stroke="#123B4F" stroke-width="4"/>
        <line x1="44" y1="56" x2="88" y2="56" stroke="#123B4F" stroke-width="4"/>
        <line x1="44" y1="72" x2="80" y2="72" stroke="#123B4F" stroke-width="4"/>
    </svg>""",

    "قطار": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <rect x="18" y="42" width="84" height="42" rx="8" fill="none" stroke="#123B4F" stroke-width="6"/>
        <line x1="18" y1="60" x2="102" y2="60" stroke="#123B4F" stroke-width="5"/>
        <circle cx="36" cy="94" r="8" fill="none" stroke="#123B4F" stroke-width="6"/>
        <circle cx="84" cy="94" r="8" fill="none" stroke="#123B4F" stroke-width="6"/>
        <line x1="30" y1="42" x2="30" y2="26" stroke="#123B4F" stroke-width="6" stroke-linecap="round"/>
    </svg>""",

    "سرير": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <rect x="14" y="60" width="92" height="30" rx="4" fill="none" stroke="#123B4F" stroke-width="6"/>
        <rect x="20" y="42" width="26" height="18" rx="4" fill="#D9A441"/>
        <line x1="14" y1="90" x2="14" y2="104" stroke="#123B4F" stroke-width="6" stroke-linecap="round"/>
        <line x1="106" y1="90" x2="106" y2="104" stroke="#123B4F" stroke-width="6" stroke-linecap="round"/>
    </svg>""",

    "غزال": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <path d="M42 100 L42 62 Q38 48 46 36 Q50 24 62 26 Q70 28 70 38 L84 34 Q80 42 72 46 Q78 52 76 62 L76 100"
              fill="none" stroke="#123B4F" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>
        <path d="M50 30 Q46 18 38 14 M62 26 Q66 14 76 12" fill="none" stroke="#123B4F" stroke-width="5" stroke-linecap="round"/>
    </svg>""",

    "جمال": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 98 Q14 70 28 66 Q34 48 44 58 Q50 42 60 56 Q68 40 78 58 Q92 62 98 90 L98 100 L12 100 Z"
              fill="none" stroke="#123B4F" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>
        <line x1="28" y1="100" x2="28" y2="110" stroke="#123B4F" stroke-width="6" stroke-linecap="round"/>
        <line x1="84" y1="100" x2="84" y2="110" stroke="#123B4F" stroke-width="6" stroke-linecap="round"/>
    </svg>""",

    "هلال": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <path d="M70 16 A38 38 0 1 0 70 96 A30 30 0 1 1 70 16 Z" fill="#D9A441"/>
        <path d="M92 30 L95 38 L103 38 L96 43 L99 51 L92 46 L85 51 L88 43 L81 38 L89 38 Z" fill="#D9A441"/>
    </svg>""",

    "بطيخ": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <path d="M14 60 A46 46 0 0 0 106 60 Z" fill="#D9A441"/>
        <path d="M14 60 A46 46 0 0 0 106 60" fill="none" stroke="#123B4F" stroke-width="6"/>
        <path d="M24 60 A36 36 0 0 0 96 60 Z" fill="#FDF6E9"/>
        <g fill="#123B4F"><circle cx="50" cy="52" r="2.6"/><circle cx="60" cy="56" r="2.6"/><circle cx="70" cy="52" r="2.6"/></g>
    </svg>""",

    "ثعلب": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <path d="M60 100 L34 60 L20 24 L46 40 L60 32 L74 40 L100 24 L86 60 Z" fill="none" stroke="#123B4F" stroke-width="6" stroke-linejoin="round"/>
        <circle cx="50" cy="58" r="3" fill="#123B4F"/><circle cx="70" cy="58" r="3" fill="#123B4F"/>
    </svg>""",

    "جدول": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <path d="M10 70 Q35 50 60 70 T110 70" fill="none" stroke="#D9A441" stroke-width="7" stroke-linecap="round"/>
        <path d="M10 88 Q35 68 60 88 T110 88" fill="none" stroke="#D9A441" stroke-width="7" stroke-linecap="round"/>
        <line x1="30" y1="50" x2="30" y2="30" stroke="#123B4F" stroke-width="5" stroke-linecap="round"/>
        <path d="M30 30 Q22 24 30 18 Q38 24 30 30" fill="#123B4F"/>
    </svg>""",

    "تمساح": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <path d="M10 78 Q30 66 50 74 Q70 60 90 70 Q100 66 108 72 L104 80 Q94 78 90 82 Q70 76 52 84 Q32 78 14 86 Z"
              fill="none" stroke="#123B4F" stroke-width="6" stroke-linejoin="round"/>
        <g fill="#123B4F"><polygon points="34,66 40,58 46,66"/><polygon points="52,64 58,56 64,64"/><polygon points="70,62 76,54 82,62"/></g>
        <circle cx="98" cy="72" r="2.5" fill="#123B4F"/>
    </svg>""",

    "برتقال": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <circle cx="60" cy="64" r="42" fill="#D9A441"/>
        <path d="M60 22 L60 12" stroke="#123B4F" stroke-width="5" stroke-linecap="round"/>
        <path d="M60 14 Q72 8 80 18" fill="none" stroke="#123B4F" stroke-width="5" stroke-linecap="round"/>
    </svg>""",

    "دولاب": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <rect x="24" y="14" width="72" height="94" rx="4" fill="none" stroke="#123B4F" stroke-width="6"/>
        <line x1="60" y1="14" x2="60" y2="108" stroke="#123B4F" stroke-width="6"/>
        <circle cx="52" cy="60" r="3" fill="#D9A441"/><circle cx="68" cy="60" r="3" fill="#D9A441"/>
    </svg>""",

    "تلفاز": """<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
        <rect x="14" y="24" width="92" height="60" rx="6" fill="none" stroke="#123B4F" stroke-width="6"/>
        <line x1="46" y1="98" x2="74" y2="98" stroke="#123B4F" stroke-width="6" stroke-linecap="round"/>
        <line x1="60" y1="84" x2="60" y2="98" stroke="#123B4F" stroke-width="6"/>
    </svg>""",
}


# ---------------------------------------------------------------------------
# تحميل النموذج
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    if not TF_AVAILABLE:
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        return None


def is_canvas_blank(image_data: np.ndarray) -> bool:
    """يتحقق إن كانت لوحة الرسم فارغة (لا يوجد رسم عليها)."""
    if image_data is None:
        return True
    gray = np.array(Image.fromarray(image_data.astype("uint8")).convert("L"))
    return gray.mean() < 3


def preprocess(image_data: np.ndarray) -> np.ndarray:
    img = Image.fromarray(image_data.astype("uint8")).convert("L").resize((32, 32))
    arr = np.array(img) / 255.0
    return arr.reshape(1, 32, 32, 1)


def predict_letter(model, image_data: np.ndarray):
    x = preprocess(image_data)
    pred = model.predict(x, verbose=0)
    idx = int(np.argmax(pred))
    confidence = float(np.max(pred))
    return LETTERS[idx], confidence


# ---------------------------------------------------------------------------
# منطق اللعبة
# ---------------------------------------------------------------------------

def new_round(level: int):
    """يبني جولة جديدة: يختار كلمة عشوائية ويحدد الحروف الناقصة."""
    bank = WORD_BANK[level]
    previous = st.session_state.get("current_word")
    choices = [w for w in bank if w != previous] or bank
    word = random.choice(choices)

    blanks_count = level
    positions = sorted(random.sample(range(len(word)), blanks_count))
    missing_letters = [word[i] for i in positions]

    st.session_state.current_word = word
    st.session_state.positions = positions
    st.session_state.missing_letters = missing_letters
    st.session_state.checked = False
    st.session_state.results = None
    st.session_state.round_id = st.session_state.get("round_id", 0) + 1


def init_state():
    defaults = {
        "level": 1,
        "score": 0,
        "correct_in_level": 0,
        "round_id": 0,
        "checked": False,
        "results": None,
        "game_complete": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "current_word" not in st.session_state:
        new_round(st.session_state.level)


def restart_game():
    for key in ["level", "score", "correct_in_level", "round_id", "checked",
                "results", "game_complete", "current_word", "positions",
                "missing_letters"]:
        st.session_state.pop(key, None)
    init_state()


def register_result(all_correct: bool):
    st.session_state.score += sum(1 for r in st.session_state.results if r["correct"])
    if all_correct:
        st.session_state.correct_in_level += 1
    if st.session_state.correct_in_level >= WORDS_TO_ADVANCE:
        if st.session_state.level < MAX_LEVEL:
            st.session_state.level += 1
            st.session_state.correct_in_level = 0
        else:
            st.session_state.game_complete = True


# ---------------------------------------------------------------------------
# التنسيق البصري (CSS)
# ---------------------------------------------------------------------------

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800;900&family=Baloo+2:wght@600;700;800&display=swap');

        :root {
            --ink: #25233a;
            --purple: #6657d9;
            --purple-dark: #44359f;
            --yellow: #ffc857;
            --orange: #ff8a4c;
            --cream: #fffaf0;
            --mint: #8de3c1;
            --pink: #ff8db5;
        }

        html, body, [class*="css"] {
            font-family: 'Cairo', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 12%, rgba(255,200,87,.30) 0 90px, transparent 91px),
                radial-gradient(circle at 92% 18%, rgba(141,227,193,.30) 0 120px, transparent 121px),
                linear-gradient(135deg, #fff8ed 0%, #f3efff 52%, #eafcff 100%);
            color: var(--ink);
        }

        #MainMenu, footer, header { visibility: hidden; }

        .block-container {
            max-width: 1180px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        .app-header {
            direction: rtl;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
            border: 3px solid rgba(255,255,255,.8);
            border-radius: 28px;
            background: linear-gradient(135deg, #6556d8, #44359f);
            box-shadow: 0 10px 0 #332778, 0 18px 35px rgba(68,53,159,.22);
            color: white;
        }

        .app-title {
            margin: 0;
            font-size: clamp(1.35rem, 3vw, 2.25rem);
            font-weight: 900;
            line-height: 1.2;
            letter-spacing: -.5px;
        }

        .app-subtitle {
            margin: .35rem 0 0;
            color: #e8e4ff;
            font-size: .95rem;
            font-weight: 600;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: .55rem;
        }

        .badge {
            min-width: 78px;
            padding: .5rem .75rem;
            border-radius: 18px;
            text-align: center;
            color: var(--ink);
            background: #fff;
            border: 3px solid #eee9ff;
            box-shadow: 0 4px 0 rgba(0,0,0,.10);
            font-size: 1rem;
            font-weight: 900;
        }

        .badge span {
            display: block;
            color: #77718f;
            font-size: .68rem;
            font-weight: 700;
        }

        [data-testid="stProgress"] > div > div {
            background: linear-gradient(90deg, var(--yellow), var(--orange), var(--pink));
            border-radius: 999px;
        }

        [data-testid="stProgress"] {
            height: 14px;
            margin: .35rem 0 1.2rem;
        }

        .game-card {
            direction: rtl;
            background: rgba(255,255,255,.92);
            border: 3px solid #fff;
            border-radius: 32px;
            padding: clamp(1rem, 3vw, 2.4rem);
            box-shadow: 0 12px 0 #ddd7ef, 0 22px 45px rgba(66,53,110,.12);
            margin-bottom: 1.4rem;
        }

        .instruction {
            text-align: center;
            color: #5f587b;
            font-size: 1.05rem;
            font-weight: 700;
            margin: .4rem 0 1.3rem;
        }

        .hint-wrapper {
            display: flex;
            justify-content: center;
            margin: .4rem 0 1rem;
        }

        .hint-card {
            width: 165px;
            height: 165px;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background: linear-gradient(145deg, #fff7d9, #fff);
            border: 4px solid var(--yellow);
            border-radius: 32px;
            box-shadow: 0 8px 0 #e6b34c, 0 14px 25px rgba(230,179,76,.20);
            transform: rotate(-2deg);
        }

        .hint-card svg {
            width: 100%;
            height: 100%;
        }

        .word-display {
            direction: rtl;
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: .65rem;
            margin: 1.4rem 0 1.8rem;
        }

        .letter-box, .blank-box {
            width: clamp(48px, 8vw, 76px);
            height: clamp(60px, 10vw, 88px);
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 20px;
            font-size: clamp(1.7rem, 4vw, 2.7rem);
            font-weight: 900;
            transition: transform .2s ease;
        }

        .letter-box {
            color: #fff;
            background: linear-gradient(145deg, #7162e6, #5144bd);
            border: 3px solid #8e82f0;
            box-shadow: 0 7px 0 #3d3195;
        }

        .blank-box {
            color: #d18a13;
            background: #fff8dc;
            border: 4px dashed #f2b93f;
            box-shadow: 0 7px 0 #e6c77c;
        }

        .canvas-label {
            direction: rtl;
            text-align: center;
            color: #554b78;
            font-weight: 900;
            margin: .5rem 0;
        }

        [data-testid="stCanvas"] {
            border: 4px solid #6657d9 !important;
            border-radius: 22px !important;
            box-shadow: 0 7px 0 #44359f, 0 12px 25px rgba(68,53,159,.15);
            overflow: hidden;
        }

        .feedback-box {
            direction: rtl;
            border-radius: 18px;
            padding: .9rem 1.1rem;
            margin: .7rem 0;
            text-align: center;
            font-weight: 800;
            border: 3px solid;
        }

        .feedback-correct {
            background: #e4fff3;
            color: #14734f;
            border-color: #8de3c1;
        }

        .feedback-wrong {
            background: #fff0f4;
            color: #b33c66;
            border-color: #ff9cbb;
        }

        .feedback-unknown {
            background: #f1effa;
            color: #625b82;
            border-color: #d4cef0;
        }

        .stButton > button {
            direction: rtl;
            width: 100%;
            min-height: 48px;
            border: 0;
            border-radius: 17px;
            font-family: 'Cairo', sans-serif;
            font-size: 1rem;
            font-weight: 900;
            transition: transform .15s ease, box-shadow .15s ease;
            box-shadow: 0 5px 0 rgba(50,40,90,.22);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 7px 0 rgba(50,40,90,.22);
        }

        .stButton > button:active {
            transform: translateY(3px);
            box-shadow: 0 2px 0 rgba(50,40,90,.22);
        }

        div[data-testid="stButton"] button[kind="primary"] {
            color: #35265d;
            background: linear-gradient(135deg, #ffd66e, #ffab52);
        }

        div[data-testid="stButton"] button[kind="secondary"] {
            color: white;
            background: linear-gradient(135deg, #7162e6, #5144bd);
        }

        .level-complete {
            direction: rtl;
            text-align: center;
            padding: 3rem 1.5rem;
            color: white;
            border: 4px solid white;
            border-radius: 32px;
            background: linear-gradient(135deg, #6657d9, #44359f);
            box-shadow: 0 10px 0 #332778, 0 20px 35px rgba(68,53,159,.2);
            font-size: 1.35rem;
            font-weight: 900;
        }

        div[data-testid="stExpander"] {
            border: 2px solid #ded7f3;
            border-radius: 18px;
            background: rgba(255,255,255,.7);
        }

        @media (max-width: 700px) {
            .app-header {
                flex-direction: column;
                text-align: center;
            }
            .badge-row {
                width: 100%;
            }
            .hint-card {
                width: 135px;
                height: 135px;
            }
            .game-card {
                border-radius: 24px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# مكونات الواجهة
# ---------------------------------------------------------------------------

def render_header():
    level = st.session_state.level
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <p class="app-title">🎮 مغامرة الحروف العربية</p>
                <p class="app-subtitle">ارسم الحروف • اجمع النجوم • افتح المراحل</p>
            </div>
            <div class="badge-row">
                <div class="badge">{LEVEL_LABELS[level]}<span>المستوى</span></div>
                <div class="badge">{st.session_state.score}<span>النقاط</span></div>
                <div class="badge">{st.session_state.correct_in_level}/{WORDS_TO_ADVANCE}<span>للترقية</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(st.session_state.correct_in_level / WORDS_TO_ADVANCE)


def render_hint_image(word: str):
    icon = WORD_ICONS.get(word)
    if not icon:
        return
    st.markdown(
        f'<div class="hint-wrapper"><div class="hint-card">{icon}</div></div>',
        unsafe_allow_html=True,
    )


def render_word_display(word: str, positions: list):
    spans = []
    for i, ch in enumerate(word):
        if i in positions:
            spans.append('<div class="blank-box">؟</div>')
        else:
            spans.append(f'<div class="letter-box">{ch}</div>')
    st.markdown(f'<div class="word-display">{"".join(spans)}</div>', unsafe_allow_html=True)


ORDINALS = ["الأول", "الثاني", "الثالث", "الرابع"]


def render_canvases(count: int, round_id: int):
    canvases = []
    cols = st.columns(count)
    for i, col in enumerate(cols):
        with col:
            st.markdown(
                f'<div class="canvas-label">الحرف الناقص {ORDINALS[i]}</div>',
                unsafe_allow_html=True,
            )
            result = st_canvas(
                stroke_width=11,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=CANVAS_SIZE,
                width=CANVAS_SIZE,
                drawing_mode="freedraw",
                return_image_data=True,
                key=f"canvas_{i}_{round_id}",
            )
            canvases.append(result)
    return canvases


def render_feedback(results):
    for i, r in enumerate(results):
        if r["status"] == "correct":
            css_class = "feedback-correct"
            text = f"الحرف {ORDINALS[i]}: إجابة صحيحة — رسمت حرف \"{r['predicted']}\""
        elif r["status"] == "wrong":
            css_class = "feedback-wrong"
            text = f"الحرف {ORDINALS[i]}: إجابة غير صحيحة — الحرف المطلوب \"{r['expected']}\""
        else:
            css_class = "feedback-unknown"
            text = f"الحرف {ORDINALS[i]}: لم يتم رسم أي شيء بعد"
        st.markdown(f'<div class="feedback-box {css_class}">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# التطبيق الرئيسي
# ---------------------------------------------------------------------------

def main():
    inject_css()
    init_state()
    model = load_model()

    render_header()

    if st.session_state.game_complete:
        st.markdown(
            f"""
            <div class="level-complete">
                🏆 أسطوري! لقد أكملت جميع المستويات بنجاح<br>
                مجموع نقاطك: {st.session_state.score}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        col = st.columns([1, 1, 1])[1]
        with col:
            if st.button("ابدأ من جديد", use_container_width=True, type="primary"):
                restart_game()
                st.rerun()
        return

    st.markdown('<div class="game-card">', unsafe_allow_html=True)

    if model is None:
        st.warning(
            "لم يتم العثور على نموذج التعرف على الحروف (arabic_letters_model.keras). "
            "ضع ملف النموذج في نفس مجلد التطبيق حتى يعمل التحقق من الإجابات.",
            icon="⚠️",
        )

    st.markdown('<p class="instruction">مهمتك: انظر للصورة، ثم ارسم الحروف الناقصة بخط يدك ✨</p>',
                unsafe_allow_html=True)

    word = st.session_state.current_word
    positions = st.session_state.positions

    render_hint_image(word)
    render_word_display(word, positions)
    canvases = render_canvases(len(positions), st.session_state.round_id)

    st.write("")
    action_cols = st.columns([1, 1, 3])

    with action_cols[0]:
        check_clicked = st.button("تحقق", type="primary", use_container_width=True)
    with action_cols[1]:
        next_clicked = st.button("كلمة جديدة", use_container_width=True,
                                  disabled=not st.session_state.checked)

    if check_clicked:
        if model is None:
            st.error("لا يمكن التحقق من الإجابة بدون تحميل النموذج.")
        else:
            results = []
            for canvas_result, expected in zip(canvases, st.session_state.missing_letters):
                image_data = canvas_result.image_data if canvas_result else None
                if image_data is None or is_canvas_blank(image_data):
                    results.append({"status": "empty", "expected": expected,
                                     "predicted": None, "correct": False})
                    continue
                predicted, _ = predict_letter(model, image_data)
                correct = predicted == expected
                results.append({
                    "status": "correct" if correct else "wrong",
                    "expected": expected,
                    "predicted": predicted,
                    "correct": correct,
                })

            st.session_state.results = results
            st.session_state.checked = True
            register_result(all(r["correct"] for r in results))
            st.rerun()

    if next_clicked:
        new_round(st.session_state.level)
        st.rerun()

    if st.session_state.checked and st.session_state.results:
        st.write("")
        render_feedback(st.session_state.results)
        if all(r["correct"] for r in st.session_state.results):
            st.success("⭐ ممتاز! الكلمة كاملة وصحيحة — كسبت نجمة!")
        else:
            st.info("💪 قربت! حاول مرة أخرى، أنت تقدر!")

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("إعادة ضبط اللعبة"):
        if st.button("إعادة تشغيل اللعبة من البداية"):
            restart_game()
            st.rerun()


if __name__ == "__main__":
    main()