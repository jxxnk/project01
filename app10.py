import streamlit as st
import datetime
import time  
import os
import base64
import joblib 
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components 

# 시연 절차
# streamlit run 파일명.py -> 새로운 터미널에서 ./ngrok http 8501 -> 나타난 터미널 화면의 Forwarding 주소 상대방이 접속하게 전달

# 모듈 임포트
from censor_runtime_interface import CensorRuntime, AttachmentTextExtractor
from build_token_features import process_message
from censor_adapter import process_message_with_llm

# 페이지 설정
st.set_page_config(page_title="SK shieldus talk", page_icon="💬", layout="wide")

# ML 모델 초기화 (앱 시작 시 1회만 로드)
@st.cache_resource
def load_censor_models():
    runtime = CensorRuntime("final_censor_model_kiwi.joblib")
    dict_vectorizer = joblib.load("dict_vectorizer_kiwi.pkl")
    return runtime, dict_vectorizer

runtime, dict_vectorizer = load_censor_models()

# 채팅방 ui 설정 (기존 앱 디자인 100% 유지)
st.markdown("""
<style>
    /* 전체 페이지 배경은 화이트와 오렌지 색상의 그라데이션 컬러로 설정 */
    .stApp {
        background: linear-gradient(180deg, #FFFFFF 0%, #FFF6F0 60%, #FFEDE0 100%) !important;
        background-attachment: fixed !important;
    }

    /* 전체 콘텐츠 폭은 wide 레이아웃에서 최대 1200px로 중앙 정렬 */
    div[data-testid="stMainBlockContainer"], .block-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding-top: 2.5rem !important;
    }

    div[data-testid="stScrollableContainer"] { background-color: #BACEE0 !important; padding: 20px !important; border-radius: 12px !important; box-shadow: inset 0px 1px 3px rgba(0, 0, 0, 0.05) !important; }
    .chat-bubble { padding: 14px 18px; border-radius: 18px; font-size: 17px; line-height: 1.5; word-break: break-word; white-space: pre-wrap; display: inline-block; box-shadow: 0px 1px 2px rgba(0, 0, 0, 0.1); }
    .user-container { display: flex; justify-content: flex-end; width: 100%; margin-bottom: 12px; }
    .user-bubble { background-color: #FEE500; color: #191919; border-top-right-radius: 3px; text-align: left; max-width: 70%; }
    .assistant-container { display: flex; justify-content: flex-start; width: 100%; margin-bottom: 12px; }
    .assistant-container > div { max-width: 70%; } 
    .assistant-bubble { background-color: #FFFFFF; color: #191919; border: 1px solid #E2E2E2; border-top-left-radius: 3px; text-align: left; max-width: 100%; }
    .msg-name { font-weight: bold; font-size: 14px; margin-bottom: 5px; color: #555; }
    .msg-time { font-size: 13px; color: #999; margin-left: 6px; margin-right: 6px; display: flex; align-items: flex-end; padding-bottom: 2px; flex-shrink: 0;}
    .sys-msg { text-align: center; color: #777; font-size: 14px; padding: 8px 16px; background-color: rgba(0,0,0,0.05); border-radius: 20px; margin: 12px auto; width: fit-content; }
    
    /* 상위 st.container가 강제로 그리는 빈 테두리와 회색 윤곽선 완벽히 박살내기 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important;
        background-color: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    /* gap 제거는 채팅 바 내부에만 적용 */
    div.st-key-chatbar div[data-testid="stVerticalBlock"] {
        gap: 0rem !important;
    }
    /* 채팅 메시지 목록의 요소 간 간격 확보 */
    div[data-testid="stVerticalBlock"] {
        gap: 1.2rem !important;
    }
    /* 메시지 요소 박스가 말풍선 실제 높이보다 작게 계산되어 겹치는 현상 방지 */
    div[data-testid="stElementContainer"],
    div[data-testid="stMarkdown"],
    div[data-testid="stMarkdown"] > div {
        min-height: fit-content !important;
        overflow: visible !important;
    }
    .user-container, .assistant-container {
        min-height: fit-content !important;
    }
    div.st-key-chatbar div[data-testid="stVerticalBlock"],
    div.st-key-chatbar div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] {
        gap: 0rem !important;
    }

    /* key="chatbar" 컨테이너 안의 가로 블록 자체를 하나의 둥근 바로 스타일링 */
    div.st-key-chatbar div[data-testid="stHorizontalBlock"] {
        background-color: #FFFFFF !important;
        border-radius: 34px !important;
        padding: 8px 18px 8px 10px !important;
        align-items: center !important;
        gap: 0rem !important;
        flex-wrap: nowrap !important;
        border: 1px solid transparent !important;
        box-shadow: 0 1px 3px rgba(60, 64, 67, 0.12), 0 1px 2px rgba(60, 64, 67, 0.24) !important;
        transition: box-shadow 0.2s, background-color 0.2s;
        margin-top: 20px !important;
    }

    /* 입력창 포커스 시 부드러운 테두리 강조 및 배경색 흰색 전환 */
    div.st-key-chatbar div[data-testid="stHorizontalBlock"]:focus-within {
        background-color: #ffffff !important;
        border-color: #e8eaed !important;
        box-shadow: 0 4px 6px rgba(32, 33, 36, 0.08), 0 1px 3px rgba(32, 33, 36, 0.16) !important;
    }

    /* 컬럼 내부 엘리먼트 가로/세로 정렬 보정 */
    div.st-key-chatbar div[data-testid="stColumn"],
    div.st-key-chatbar div[data-testid="column"] {
        display: flex !important;
        align-items: center !important;
        background-color: transparent !important;
        min-width: 0 !important;
    }
    /* + 버튼 컬럼이 찌그러지지 않도록 최소 폭 확보 */
    div.st-key-chatbar div[data-testid="stColumn"]:first-child,
    div.st-key-chatbar div[data-testid="column"]:first-child {
        flex: 0 0 58px !important;
        justify-content: center !important;
    }

    /* 팝오버 '+' 버튼 영역 투명화 및 테두리 완벽 제거 */
    div.st-key-chatbar div[data-testid="stPopover"] {
        background-color: transparent !important;
        border: none !important;
    }
    /* 버튼이 stPopover의 직계 자식이 아닌 버전도 있으므로 후손 셀렉터로 전부 매칭 */
    div.st-key-chatbar div[data-testid="stPopover"] button {
        height: 48px !important; 
        width: 48px !important;
        min-width: 48px !important;
        border-radius: 50% !important; 
        padding: 0 !important; 
        font-size: 26px !important; 
        border: none !important;
        outline: none !important;
        background-color: transparent !important; 
        color: #444746 !important; 
        box-shadow: none !important;
        cursor: pointer !important;
        transition: background-color 0.15s ease, box-shadow 0.15s ease, color 0.15s ease;
    }
    /* 팝오버 버튼 오른쪽에 붙는 기본 화살표(∨) 아이콘 제거 */
    div.st-key-chatbar div[data-testid="stPopover"] button svg,
    div.st-key-chatbar div[data-testid="stPopover"] button [data-testid="stIconMaterial"] {
        display: none !important;
    }
    /* 마우스를 올리면 원형 음영과 얇은 테두리 링  */
    div.st-key-chatbar div[data-testid="stPopover"] button:hover { 
        background-color: #E2E6EB !important; 
        color: #1f1f1f !important; 
        box-shadow: inset 0 0 0 1px rgba(60, 64, 67, 0.18) !important;
    }
    /* 클릭 시 진한 음영을 추가 */
    div.st-key-chatbar div[data-testid="stPopover"] button:active { 
        background-color: #D3D7DC !important; 
    }
    /* 포커스 시 남는 빨간/회색 포커스 링 제거 */
    div.st-key-chatbar div[data-testid="stPopover"] button:focus,
    div.st-key-chatbar div[data-testid="stPopover"] button:focus-visible { 
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    
    /* 텍스트 입력창 고유의 회색 배경 및 테두리 영역 강제 무력화 */
    div.st-key-chatbar div[data-testid="stTextInput"] {
        width: 100% !important;
        background-color: transparent !important;
        border: none !important;
    }
    div.st-key-chatbar div[data-testid="stTextInput"] div[data-testid="InputRootElement"],
    div.st-key-chatbar div[data-testid="stTextInput"] > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    div.st-key-chatbar input {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding-left: 10px !important;
        font-size: 17px !important;
        height: 3rem !important;
        color: #191919 !important;
    }

    /* 채팅 바 내부 st.form: 기본 테두리와 패딩을 제거하고 투명하게 */
    div.st-key-chatbar div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        background-color: transparent !important;
        box-shadow: none !important;
        width: 100% !important;
    }
    /* 폼 내부의 중첩 가로 블록에는 pill 스타일이 중복 적용되지 않도록 초기화 */
    div.st-key-chatbar div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
        background-color: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin-top: 0 !important;
        gap: 0rem !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
    }
    /* 폼 내부 첫 번째 컬럼은 +버튼용 58px 고정폭 규칙에 걸리지 않도록 복원 */
    div.st-key-chatbar div[data-testid="stForm"] div[data-testid="stColumn"]:first-child,
    div.st-key-chatbar div[data-testid="stForm"] div[data-testid="column"]:first-child {
        flex: 1 1 auto !important;
        justify-content: flex-start !important;
    }
    /* 전송 버튼 컬럼은 고정폭 */
    div.st-key-chatbar div[data-testid="stForm"] div[data-testid="stColumn"]:last-child,
    div.st-key-chatbar div[data-testid="stForm"] div[data-testid="column"]:last-child {
        flex: 0 0 52px !important;
        justify-content: center !important;
    }
    /* 채팅 전송 버튼 디자인 */
    div.st-key-chatbar div[data-testid="stFormSubmitButton"] button {
        height: 40px !important;
        width: 40px !important;
        min-width: 40px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        border: none !important;
        background: linear-gradient(135deg, #EA002C 0%, #F47725 100%) !important;
        color: #FFFFFF !important;
        font-size: 16px !important;
        cursor: pointer !important;
        transition: filter 0.15s ease, box-shadow 0.15s ease;
    }
    div.st-key-chatbar div[data-testid="stFormSubmitButton"] button:hover {
        filter: brightness(1.08);
        box-shadow: 0 3px 10px rgba(234, 0, 44, 0.30) !important;
    }
    div.st-key-chatbar div[data-testid="stFormSubmitButton"] button:disabled {
        background: #E2E6EB !important;
        color: #9AA0A6 !important;
        cursor: not-allowed !important;
    }
    /* 입력창 포커스 시 뜨는 'Press Enter to submit form' 안내 문구 숨김 */
    div.st-key-chatbar [data-testid="InputInstructions"] {
        display: none !important;
    }

    /* DLP 전송 방식 선택 버튼 */
    div.st-key-dlp-warning div[data-testid="stButton"] button {
        background-color: #FFFFFF !important;
        border: 1px solid #E9E4E0 !important;
        border-radius: 12px !important;
        height: 48px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #191919 !important;
        box-shadow: 0 1px 3px rgba(60, 64, 67, 0.10) !important;
        transition: border-color 0.2s, box-shadow 0.2s, background-color 0.2s;
    }
    div.st-key-dlp-warning div[data-testid="stButton"] button:hover {
        border-color: #F47725 !important;
        box-shadow: 0 0 0 3px rgba(244, 119, 37, 0.12) !important;
    }
    /* 알림 블록 간격이 전역 gap(1.2rem)으로 너무 벌어지지 않도록 조정 */
    div.st-key-dlp-warning div[data-testid="stVerticalBlock"] {
        gap: 0.6rem !important;
    }
</style>
""", unsafe_allow_html=True)

# 공용 데이터베이스
@st.cache_resource
def get_global_chat_history():
    return []

chat_db = get_global_chat_history()

# 세션 상태 관리 초기화
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "user_id" not in st.session_state: st.session_state.user_id = ""
if "has_joined" not in st.session_state: st.session_state.has_joined = False
if "join_timestamp" not in st.session_state: st.session_state.join_timestamp = 0.0 

if "just_sent" not in st.session_state: st.session_state.just_sent = False
if "show_dlp_warning" not in st.session_state: st.session_state.show_dlp_warning = False

# [추가] 파일 전송 기능을 텍스트 처리와 병합하기 위한 세션 상태
if "is_processing" not in st.session_state: st.session_state.is_processing = False
if "temp_input" not in st.session_state: st.session_state.temp_input = ""
if "flagged_spans" not in st.session_state: st.session_state.flagged_spans = []
if "is_pending_file" not in st.session_state: st.session_state.is_pending_file = False
if "pending_file_name" not in st.session_state: st.session_state.pending_file_name = ""

# 채팅 종료 후 평가 화면 관리를 위한 세션 상태
if "show_eval" not in st.session_state: st.session_state.show_eval = False
if "eval_done" not in st.session_state: st.session_state.eval_done = False
if "eval_user" not in st.session_state: st.session_state.eval_user = ""
if "pending_msg" not in st.session_state: st.session_state.pending_msg = ""
if "masked_msg" not in st.session_state: st.session_state.masked_msg = ""

def save_audit_log(user_id, original_text, flagged_spans=None):
    if flagged_spans is None:
        flagged_spans = []
        
    log_file = "dlp_override_audit.log"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if flagged_spans:
        reason_str = ", ".join([f"{item.get('target', '알수없음')}({item.get('reason', '알수없음')})" for item in flagged_spans])
    else:
        reason_str = "사유 없음(수동 전송)"
        
    log_entry = f"[{current_time}] [WARNING] User: {user_id} | Action: DLP_OVERRIDE | Detected: {reason_str} | Original Message: {original_text}\n"
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

def save_eval_log(user_id, choice_text):
    log_file = "chat_evaluation_log.txt"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{current_time}] [EVALUATION] User: {user_id} | Choice: {choice_text}\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

# [수정] 파일과 텍스트를 동일하게 저장할 수 있도록 수정된 commit_message
def commit_message(content, is_file=False, file_name="", file_data=None):
    msg = {
        "name": st.session_state.user_id, 
        "time": datetime.datetime.now().strftime("%H:%M"),
        "timestamp": datetime.datetime.now().timestamp() 
    }
    
    # 텍스트 대신 파일 형태로 DB에 저장
    if is_file:
        msg["content"] = f"📁 **{file_name}**"
        msg["is_file"] = True
        msg["file_name"] = file_name
        msg["file_data"] = file_data
        msg["file_type"] = "text/plain"
    else:
        msg["content"] = content

    chat_db.append(msg)
    
    # 상태 싹 다 초기화
    st.session_state.show_dlp_warning = False
    st.session_state.pending_msg = ""
    st.session_state.masked_msg = ""
    st.session_state.flagged_spans = []
    st.session_state.is_pending_file = False
    st.session_state.pending_file_name = ""
    st.rerun()

# [수정] 오직 텍스트만 받아서 LLM으로 정밀 검사하는 단일 파이프라인으로 통합 (파일은 UI 단에서 텍스트로 합쳐서 들어옴)
def security_check(text=""):
    if text:
        ml_result = process_message(text, dict_vectorizer, runtime, threshold=0.85, mode="mask")
        
        if ml_result["action"] == "block":
            return False, f"🚨 **[시스템 차단]** {ml_result['display_message']}", None, ml_result.get("flagged_spans", [])
            
        elif ml_result["action"] == "mask":
            raw_spans = ml_result.get("flagged_spans", [])
            safe_ml_findings = []
            
            for item in raw_spans:
                target_word = item.get("target", "") if isinstance(item, dict) else str(item)
                label_name = item.get("label", "UNKNOWN") if isinstance(item, dict) else "UNKNOWN"
                start_idx = text.find(target_word)
                if start_idx != -1 and target_word:
                    safe_ml_findings.append({
                        "start": start_idx,
                        "end": start_idx + len(target_word),
                        "label": label_name
                    })
            
            try:
                llm_result = process_message_with_llm(
                    original_text=text,
                    ml_findings=safe_ml_findings,
                    regex_findings=[], 
                    mode="mask"
                )
                
                if llm_result["action"] == "pass":
                    return True, "통과", text, []
                else:
                    return True, "⚠️ 일부 민감정보가 LLM 검토 후 마스킹 처리되었습니다.", llm_result["send_text"], llm_result.get("flagged_spans", [])
                    
            except Exception as e:
                return False, f"🚨 LLM 연동 에러 발생 (원인: {e})", None, []
                
        else:
            return True, "통과", text, []
            
    return True, "통과", "", []

# 채팅 종료 후 평가 화면 
if st.session_state.show_eval:
    # 평가 화면 전용 CSS 
    st.markdown("""
    <style>
        div.st-key-eval-card {
            background-color: #FFFFFF;
            border: 1px solid #F3E7DF;
            border-radius: 24px;
            padding: 60px 64px 48px 64px;
            margin-top: 10vh;
            box-shadow: 0 14px 44px rgba(234, 0, 44, 0.07), 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        div.st-key-eval-card img {
            display: block;
            margin: 0 auto 6px auto;
        }
        .eval-title {
            text-align: center;
            color: #191919;
            font-size: 30px;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin: 20px 0 12px 0;
        }
        .eval-title .accent { color: #EA002C; }
        .eval-sub {
            text-align: center;
            color: #8A8F98;
            font-size: 16px;
            line-height: 1.8;
            margin-bottom: 30px;
        }
        div.st-key-eval-card div[data-testid="stRadio"] label,
        div.st-key-eval-card div[data-testid="stCheckbox"] label {
            background-color: #FAFAFA;
            border: 1px solid #E9E4E0;
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 12px;
            width: 100%;
            transition: border-color 0.2s, background-color 0.2s, box-shadow 0.2s;
        }
        div.st-key-eval-card div[data-testid="stRadio"] label:hover,
        div.st-key-eval-card div[data-testid="stCheckbox"] label:hover {
            background-color: #FFFFFF;
            border-color: #F47725;
            box-shadow: 0 0 0 3px rgba(244, 119, 37, 0.12);
        }
        div.st-key-eval-card div[data-testid="stRadio"] label p,
        div.st-key-eval-card div[data-testid="stCheckbox"] label p {
            font-size: 16px !important;
            color: #191919 !important;
            line-height: 1.5;
        }
        div.st-key-eval-card div[data-testid="stCheckbox"] {
            margin-bottom: 0px;
        }
        div.st-key-eval-card div[data-testid="stButton"] button {
            width: 100% !important;
            height: 58px !important;
            margin-top: 14px;
            border: none !important;
            border-radius: 12px !important;
            background: linear-gradient(90deg, #EA002C 0%, #F47725 100%) !important;
            color: #FFFFFF !important;
            font-size: 17px !important;
            font-weight: 700 !important;
            letter-spacing: 0.5px;
            cursor: pointer !important;
            transition: box-shadow 0.2s, transform 0.1s, filter 0.2s;
        }
        div.st-key-eval-card div[data-testid="stButton"] button:hover {
            filter: brightness(1.06);
            box-shadow: 0 6px 18px rgba(234, 0, 44, 0.25) !important;
        }
        div.st-key-eval-card div[data-testid="stButton"] button:active {
            transform: translateY(1px);
        }
        .eval-thanks {
            text-align: center;
            color: #191919;
            font-size: 26px;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin: 26px 0 10px 0;
        }
        .eval-thanks-sub {
            text-align: center;
            color: #8A8F98;
            font-size: 15px;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    if os.path.exists("sk_shieldus_logo.png"):
        with open("sk_shieldus_logo.png", "rb") as _f:
            _logo_b64 = base64.b64encode(_f.read()).decode()
        st.markdown(
            "<style>"
            ".stApp::before {"
            "  content: '';"
            "  position: fixed;"
            "  inset: 0;"
            "  background: url(data:image/png;base64," + _logo_b64 + ") no-repeat right -4% bottom -6% / 42% auto;"
            "  opacity: 0.06;"
            "  pointer-events: none;"
            "  z-index: 0;"
            "}"
            "</style>",
            unsafe_allow_html=True,
        )

    ecol1, ecol2, ecol3 = st.columns([1, 1.6, 1])
    with ecol2:
        with st.container(key="eval-card"):
            if os.path.exists("sk_shieldus_logo.png"):
                elogo_l, elogo_c, elogo_r = st.columns([1, 2, 1])
                with elogo_c:
                    st.image("sk_shieldus_logo.png", width=240)

            if not st.session_state.eval_done:
                st.markdown(
                    "<div class='eval-title'>채팅 서비스 <span class='accent'>평가</span></div>"
                    "<div class='eval-sub'>이번 채팅에서의 보안(DLP) 동작에 대해 평가해주세요.<br>"
                    "아래 항목은 중복 선택이 가능합니다.</div>",
                    unsafe_allow_html=True,
                )

                eval_options = [
                    "1. 학습된 민감 데이터가 어떠한 제재없이 출력되었습니다.",
                    "2. 학습된 민감 데이터가 유출되지 않도록 적절한 경고 조치가 이루어졌습니다.",
                    "3. 문제없는 채팅도 민감 데이터로 분류되어 채팅이 원활하지 못했습니다.",
                ]
                selected_choices = []
                for i, option in enumerate(eval_options):
                    if st.checkbox(option, key=f"eval_check_{i}"):
                        selected_choices.append(option)

                if st.button("평가 제출", use_container_width=True, key="eval_submit"):
                    if not selected_choices:
                        st.warning("평가 항목을 선택해주세요.")
                    else:
                        save_eval_log(st.session_state.eval_user, " | ".join(selected_choices))
                        st.session_state.eval_done = True
                        st.rerun()
            else:
                st.markdown(
                    "<div class='eval-thanks'>평가해주셔서 감사합니다.</div>"
                    "<div class='eval-thanks-sub'>소중한 의견은 보안 서비스 개선에 활용됩니다.</div>",
                    unsafe_allow_html=True,
                )
                time.sleep(2)
                st.session_state.show_eval = False
                st.session_state.eval_done = False
                st.session_state.eval_user = ""
                for i in range(3):
                    st.session_state.pop(f"eval_check_{i}", None)
                st.rerun()

# 로그인 화면 ui
elif not st.session_state.logged_in:
    st.markdown("""
    <style>
        div.st-key-login-card {
            background-color: #FFFFFF;
            border: 1px solid #F3E7DF;
            border-radius: 24px;
            padding: 60px 64px 48px 64px;
            margin-top: 6vh;
            box-shadow: 0 14px 44px rgba(234, 0, 44, 0.07), 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        div.st-key-login-card div[data-testid="stForm"] {
            border: none !important;
            padding: 0 !important;
            box-shadow: none !important;
        }
        div.st-key-login-card img {
            display: block;
            margin: 0 auto 6px auto;
        }
        .login-title {
            text-align: center;
            color: #191919;
            font-size: 34px;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin: 20px 0 12px 0;
        }
        .login-title .accent { color: #EA002C; }
        .login-sub {
            text-align: center;
            color: #8A8F98;
            font-size: 16px;
            line-height: 1.8;
            margin-bottom: 34px;
        }
        div.st-key-login-card div[data-testid="stTextInput"] label p {
            font-size: 15px !important;
            color: #5f6368 !important;
            font-weight: 600;
        }
        div.st-key-login-card div[data-testid="stTextInput"] div[data-baseweb="input"] {
            background-color: #FAFAFA !important;
            border: 1px solid #E9E4E0 !important;
            border-radius: 12px !important;
            transition: border-color 0.2s, box-shadow 0.2s, background-color 0.2s;
        }
        div.st-key-login-card div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
            background-color: #FFFFFF !important;
            border-color: #F47725 !important;
            box-shadow: 0 0 0 3px rgba(244, 119, 37, 0.15) !important;
        }
        div.st-key-login-card div[data-testid="stTextInput"] input {
            font-size: 16.5px !important;
            color: #191919 !important;
            padding-top: 0.7rem !important;
            padding-bottom: 0.7rem !important;
        }
        div.st-key-login-card div[data-testid="stFormSubmitButton"] button {
            width: 100% !important;
            height: 58px !important;
            margin-top: 14px;
            border: none !important;
            border-radius: 12px !important;
            background: linear-gradient(90deg, #EA002C 0%, #F47725 100%) !important;
            color: #FFFFFF !important;
            font-size: 17px !important;
            font-weight: 700 !important;
            letter-spacing: 0.5px;
            cursor: pointer !important;
            transition: box-shadow 0.2s, transform 0.1s, filter 0.2s;
        }
        div.st-key-login-card div[data-testid="stFormSubmitButton"] button:hover {
            filter: brightness(1.06);
            box-shadow: 0 6px 18px rgba(234, 0, 44, 0.25) !important;
        }
        div.st-key-login-card div[data-testid="stFormSubmitButton"] button:active {
            transform: translateY(1px);
        }
        .login-badge {
            text-align: center;
            color: #9AA0A6;
            font-size: 14px;
            margin-top: 28px;
            padding: 7px 14px;
            background-color: #FAF7F5;
            border: 1px solid #F0EAE5;
            border-radius: 20px;
            width: fit-content;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """, unsafe_allow_html=True)

    if os.path.exists("sk_shieldus_logo.png"):
        with open("sk_shieldus_logo.png", "rb") as _f:
            _logo_b64 = base64.b64encode(_f.read()).decode()
        st.markdown(
            "<style>"
            ".stApp::before {"
            "  content: '';"
            "  position: fixed;"
            "  inset: 0;"
            "  background: url(data:image/png;base64," + _logo_b64 + ") no-repeat right -4% bottom -6% / 42% auto;"
            "  opacity: 0.06;"
            "  pointer-events: none;"
            "  z-index: 0;"
            "}"
            "</style>",
            unsafe_allow_html=True,
        )

    lcol1, lcol2, lcol3 = st.columns([1, 1.6, 1])
    with lcol2:
        with st.container(key="login-card"):
            if os.path.exists("sk_shieldus_logo.png"):
                logo_l, logo_c, logo_r = st.columns([1, 2, 1])
                with logo_c:
                    st.image("sk_shieldus_logo.png", width=240)
            st.markdown(
                "<div class='login-title'>SK Shieldus <span class='accent'>Talk</span></div>"
                "<div class='login-sub'>고객의 안심과 사회의 안전을 위해 함께 고민하고<br>"
                "유연하게 소통하는 SK쉴더스 크루들의 공간입니다.</div>",
                unsafe_allow_html=True,
            )

            with st.form("login_form"):
                user_id = st.text_input("사용자 ID (닉네임)", placeholder="아이디를 입력하세요")
                user_pw = st.text_input("비밀번호", type="password", placeholder="비밀번호를 입력하세요")
                login_button = st.form_submit_button("로그인")

                if login_button:
                    if user_id and user_pw == "3333":
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id

                        if not st.session_state.has_joined:
                            st.session_state.join_timestamp = datetime.datetime.now().timestamp()
                            chat_db.append({"type": "system", "content": f"🚪 {user_id}님이 입장하셨습니다.", "timestamp": datetime.datetime.now().timestamp()})
                            st.session_state.has_joined = True

                        st.success(f"{user_id}님, 환영합니다!")
                        st.rerun()
                    else:
                        st.error("비밀번호가 일치하지 않거나 ID가 비어있습니다.")

            st.markdown("<div class='login-badge'>🔒 사내망 엔드포인트 보안(DLP) 실시간 보호 중</div>", unsafe_allow_html=True)

# 실시간 채팅방 화면 ui
else:
    # DLP 알림창이 떠 있거나 AI가 검토 중일 때는 자동 새로고침을 일시 중지
    if not st.session_state.show_dlp_warning and not st.session_state.is_processing:
        st_autorefresh(interval=2000, key="chat_refresh")

    col1, col2 = st.columns([8, 2])
    with col1:
        st.markdown(f"## 💬 SK shieldus talk")
        st.caption(f"🔒 사내망 엔드포인트 보안(DLP) 실시간 작동 중 (접속자: {st.session_state.user_id})")
    with col2:
        st.write("") 
        if st.button("로그아웃"):
            chat_db.append({"type": "system", "content": f"👋 {st.session_state.user_id}님이 퇴장하셨습니다.", "timestamp": datetime.datetime.now().timestamp()})
            st.session_state.show_eval = True
            st.session_state.eval_user = st.session_state.user_id
            st.session_state.logged_in = False
            st.session_state.user_id = ""
            st.session_state.has_joined = False
            st.session_state.join_timestamp = 0
            st.rerun()
            
    st.divider()

    with st.container(height=650, border=False):
        for idx, msg in enumerate(chat_db):
            if msg.get("timestamp", 0) < st.session_state.join_timestamp: continue
            if msg.get("type") == "system":
                st.markdown(f"<div class='sys-msg'>{msg['content']}</div>", unsafe_allow_html=True)
                continue 

            is_mine = (msg.get("name") == st.session_state.user_id)
            
            if is_mine:
                html = f'<div class="user-container"><span class="msg-time">{msg.get("time")}</span><div class="chat-bubble user-bubble">{msg.get("content")}</div></div>'
                st.markdown(html, unsafe_allow_html=True)
                if msg.get("is_file"):
                    c1, c2, c3 = st.columns([6, 3, 1])
                    with c2: st.download_button("⬇️ 다운로드", data=msg["file_data"], file_name=msg["file_name"], mime=msg.get("file_type"), key=f"dl_{idx}")
            else:
                html = f'<div class="assistant-container"><div><div class="msg-name">{msg.get("name")}</div><div style="display:flex;"><div class="chat-bubble assistant-bubble">{msg.get("content")}</div><span class="msg-time">{msg.get("time")}</span></div></div></div>'
                st.markdown(html, unsafe_allow_html=True)
                if msg.get("is_file"):
                    c1, c2, c3 = st.columns([1, 3, 6])
                    with c2: st.download_button("⬇️ 다운로드", data=msg["file_data"], file_name=msg["file_name"], mime=msg.get("file_type"), key=f"dl_{idx}")

        st.markdown("<div id='chat-end'></div>", unsafe_allow_html=True)
        js_code = f"""
        <script>
            setTimeout(function() {{
                const marker = window.parent.document.getElementById('chat-end');
                if (marker) {{
                    let el = marker.parentElement;
                    while (el && el.tagName !== 'BODY') {{
                        const css = window.parent.getComputedStyle(el);
                        if (css.overflowY === 'auto' || css.overflowY === 'scroll' || el.style.height === '650px') {{
                            el.style.setProperty('background-color', '#BACEE0', 'important');
                            el.style.setProperty('padding', '20px', 'important');
                            el.style.setProperty('border-radius', '12px', 'important');
                            el.style.setProperty('box-shadow', 'inset 0px 1px 3px rgba(0,0,0,0.05)', 'important');
                            break;
                        }}
                        el = el.parentElement;
                    }}
                    marker.scrollIntoView({{ behavior: 'smooth', block: 'end' }});
                }}
            }}, 100);
        </script>
        """
        components.html(js_code, height=0)

    # 마스킹 발생 시 사용자에게 보여줄 경고 및 전송 방식 선택 UI
    if st.session_state.show_dlp_warning:
        with st.container(key="dlp-warning"):
            st.warning("⚠️ **보안 정책 알림:** 메시지 또는 파일에 민감 정보가 포함되어 있습니다. 전송 방식을 선택해주세요.")
            st.info(f"**원본:**\n{st.session_state.pending_msg}\n\n**마스킹 적용 시:**\n{st.session_state.masked_msg}")

            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button("✅ 마스킹하여 전송", use_container_width=True):
                    # [추가] 파일 전송 시 파일 형태로 DB에 저장
                    if st.session_state.is_pending_file:
                        masked_filename = f"masked_{st.session_state.pending_file_name}"
                        commit_message("", is_file=True, file_name=masked_filename, file_data=st.session_state.masked_msg.encode('utf-8'))
                    else:
                        commit_message(st.session_state.masked_msg)
            with btn_col2:
                if st.button("🚨 원본 전송 (로그 저장)", use_container_width=True):
                    save_audit_log(st.session_state.user_id, st.session_state.pending_msg, st.session_state.flagged_spans)
                    if st.session_state.is_pending_file:
                        commit_message("", is_file=True, file_name=st.session_state.pending_file_name, file_data=st.session_state.pending_msg.encode('utf-8'))
                    else:
                        commit_message(st.session_state.pending_msg)
            with btn_col3:
                if st.button("❌ 취소", use_container_width=True):
                    st.session_state.show_dlp_warning = False
                    st.session_state.pending_msg = ""
                    st.session_state.masked_msg = ""
                    st.session_state.flagged_spans = []
                    st.session_state.is_pending_file = False
                    st.session_state.pending_file_name = ""
                    st.rerun()

    # AI 처리 중 안내 메시지 UI
    if st.session_state.is_processing:
        st.info("🤖 **보안 AI 작동 중:** 문맥을 정밀 분석하고 있습니다... (약 3~5초 소요)")

    # 커스터마이징한 채팅 바 구현
    with st.container(key="chatbar"):
        bar_col1, bar_col2 = st.columns([1, 12], vertical_alignment="center")
        
        with bar_col1:
            with st.popover("＋", use_container_width=True):
                uploaded_file = st.file_uploader("전송할 문서나 데이터를 업로드하세요", label_visibility="collapsed")
                
                # [수정] 파일 전송 시 파일명과 내용을 텍스트로 합쳐서 LLM 단일 파이프라인에 탑승
                if st.button("파일 전송하기", use_container_width=True) and uploaded_file:
                    try:
                        file_bytes = uploaded_file.getvalue()
                        file_name = uploaded_file.name
                        if file_name.lower().endswith(".txt"):
                            extracted_text = AttachmentTextExtractor.from_txt(file_bytes)
                        else:
                            extracted_text = file_bytes.decode('utf-8', errors='ignore')

                        combined_text = f"[파일명: {file_name}]\n\n{extracted_text}"

                        st.session_state.temp_input = combined_text
                        st.session_state.is_processing = True
                        st.session_state.is_pending_file = True
                        st.session_state.pending_file_name = file_name
                        st.rerun()
                    except Exception as e:
                        st.error(f"🚨 파일 추출 오류: {e}")

        with bar_col2:
            with st.form("chat_form", clear_on_submit=True, border=False):
                in_col, send_col = st.columns([12, 1], vertical_alignment="center")
                with in_col:
                    typed_text = st.text_input(
                        "메시지 입력",
                        placeholder="분석을 기다리거나 전송 방식을 선택해주세요..." if (st.session_state.show_dlp_warning or st.session_state.is_processing) else "메시지를 입력하세요 (엔터를 누르면 전송됩니다)",
                        key="chat_form_input",
                        label_visibility="collapsed",
                        disabled=st.session_state.show_dlp_warning or st.session_state.is_processing,
                    )
                with send_col:
                    submitted = st.form_submit_button(
                        "➤",
                        use_container_width=True,
                        disabled=st.session_state.show_dlp_warning or st.session_state.is_processing,
                    )

    # 텍스트 채팅 엔터 감지 시 AI 파이프라인 탑승
    if submitted and not (st.session_state.show_dlp_warning or st.session_state.is_processing):
        text = (typed_text or "").strip()
        if text:
            st.session_state.temp_input = text
            st.session_state.is_processing = True
            st.session_state.is_pending_file = False
            st.rerun()

    # [핵심] 타이머 간섭을 피하는 2단계 LLM 안전 검사 (채팅, 파일 모두 여기를 통과함)
    if st.session_state.is_processing and st.session_state.temp_input:
        saved_input = st.session_state.temp_input
        is_safe, warning_msg, final_text, flagged_spans = security_check(text=saved_input)

        st.session_state.is_processing = False
        st.session_state.temp_input = ""

        if is_safe:
            if warning_msg != "통과":
                st.session_state.show_dlp_warning = True
                st.session_state.pending_msg = saved_input
                st.session_state.masked_msg = final_text
                st.session_state.flagged_spans = flagged_spans 
                st.rerun()
            else:
                st.session_state.just_sent = True 
                
                # 기밀이 없어 바로 통과할 때, 원래 파일이었으면 다시 파일로 만들어서 전송
                if st.session_state.is_pending_file:
                    commit_message("", is_file=True, file_name=st.session_state.pending_file_name, file_data=saved_input.encode('utf-8'))
                else:
                    commit_message(final_text) 
        else:
            st.error(warning_msg)
            st.toast("메시지 전송이 보안 정책에 의해 차단되었습니다.", icon="🚫")
            time.sleep(1)
            st.rerun()

    # 전송 직후의 재실행에서만 새 입력창에 자동 포커스 
    if st.session_state.get("just_sent"):
        st.session_state.just_sent = False
        components.html("""
        <script>
            setTimeout(function() {
                let inp = window.parent.document.querySelector('.st-key-chatbar input');
                if (!inp) {
                    const all = window.parent.document.querySelectorAll('input[type="text"]');
                    if (all.length > 0) inp = all[all.length - 1];
                }
                if (inp) inp.focus();
            }, 200);
        </script>
        """, height=0)