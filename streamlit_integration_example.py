"""
streamlit_integration_example.py
===================================
Streamlit 팀에게 전달할 "이렇게 쓰면 됩니다" 예제.
실제 Streamlit UI/GPT 연동 코드는 프론트엔드 팀이 작성하지만,
아래 process_message() 호출 패턴만 그대로 따라하면 요청하신 동작이 나온다.

준비물 (같은 폴더에 있어야 함):
  - final_censor_model_kiwi.joblib   (재학습된 최종 모델)
  - dict_vectorizer_kiwi.pkl         (재학습된 새 벡터라이저)
  - censor_ensemble_model.py         (joblib 로드에 필수)
  - censor_runtime_interface.py      (실행 인터페이스)
  - build_token_features.py         (이 파일, Kiwi 토큰화 + 판정 + 마스킹)

설치: pip install kiwipiepy
"""
import joblib
from censor_runtime_interface import CensorRuntime
from build_token_features import process_message, DEFAULT_THRESHOLD


def demo():
    print("모델 로드 중...")
    runtime = CensorRuntime("final_censor_model_kiwi.joblib")
    dict_vectorizer = joblib.load("dict_vectorizer_kiwi.pkl")
    print("로드 완료\n")

    examples = [
        "카드번호는 0000-0000-0000-0000입니다.",
        "카드 번호는 0000000000000000입니다.",
        "오늘 점심 메뉴는 뭔가요?",
    ]

    print("=" * 60)
    print('모드 1: "block" -- 민감정보 있으면 전송 자체를 막음')
    print("=" * 60)
    for text in examples:
        result = process_message(text, dict_vectorizer, runtime,
                                  threshold=DEFAULT_THRESHOLD, mode="block")
        print(f"입력: {text}")
        print(f"동작: {result['action']}")
        print(f"화면 표시: {result['display_message']}")
        print()

    print("=" * 60)
    print('모드 2: "mask" -- 민감정보 부분만 가리고 전송은 허용')
    print("=" * 60)
    for text in examples:
        result = process_message(text, dict_vectorizer, runtime,
                                  threshold=DEFAULT_THRESHOLD, mode="mask")
        print(f"입력: {text}")
        print(f"동작: {result['action']}")
        print(f"화면 표시: {result['display_message']}")
        print()


# --- Streamlit 팀이 실제로 붙일 위치 예시 (의사코드) ---
"""
import streamlit as st

if st.button("보내기"):
    result = process_message(
        user_input, dict_vectorizer, runtime,
        threshold=0.85,
        mode="block",   # 또는 "mask" -- 회사 정책에 따라 선택
    )

    if result["action"] == "block":
        st.error(result["display_message"])
        # 전송/GPT 호출 하지 않음

    elif result["action"] == "mask":
        st.warning("민감정보가 감지되어 일부가 가려졌습니다.")
        st.write(result["display_message"])
        send_to_messenger(result["send_text"])   # 마스킹된 텍스트로 전송
        call_gpt(result["send_text"])

    else:  # "pass"
        send_to_messenger(result["send_text"])
        call_gpt(result["send_text"])

    # 첨부파일도 동일 -- AttachmentTextExtractor 로 텍스트 추출 후 위와 같은 흐름
    # from censor_runtime_interface import AttachmentTextExtractor
    # text = AttachmentTextExtractor.from_pdf(file_bytes)  # 또는 from_txt/from_csv
    # result = process_message(text, dict_vectorizer, runtime, threshold=0.85, mode="block")
"""


if __name__ == "__main__":
    demo()
