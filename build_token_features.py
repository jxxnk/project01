"""
build_token_features.py (최종본)
==================================
Kiwi로 형태소 분석 -> dict_vectorizer_kiwi.pkl -> final_censor_model_kiwi.joblib
로 이어지는 전체 파이프라인을 하나의 함수(process_message)로 감싼 최종본.

검증 결과(threshold 비교) 기준으로 threshold=0.85 를 기본값으로 함:
  - NORMAL 메시지 통과율 87%
  - 이진 precision 0.946 / recall 0.762
  - CREDENTIAL_SECRET 라벨은 아직 정확도가 낮아(precision 0.10~0.26),
    "이 카테고리다"라고 화면에 확정적으로 보여주면 안 됨 (아래 안내 참고)

설치: pip install kiwipiepy
"""
import re
from typing import Callable, List, Dict, Any, Literal

from kiwipiepy import Kiwi

try:
    from kiwipiepy import Match
    _MATCH_ALL = Match.ALL
except ImportError:
    _MATCH_ALL = None

_kiwi = Kiwi()

# 검증 결과 기준 권장 threshold. Streamlit 팀에서 운영 데이터 쌓이면 재조정 가능.
DEFAULT_THRESHOLD = 0.85

# CREDENTIAL_SECRET 은 precision이 매우 낮게 측정됨(0.10~0.26) -- 이 라벨이
# top_labels 로 나와도 화면에 "인증정보(비밀번호/API키)입니다" 라고 단정적으로
# 보여주지 말고, 그냥 "민감정보로 추정됨" 정도로만 안내할 것을 권장.
LOW_CONFIDENCE_LABELS = {"CREDENTIAL_SECRET"}


# --- 1. Kiwi 토큰화 (span 포함) ---
def tokenize_with_span(text: str) -> List[Dict[str, Any]]:
    kwargs = {"match_options": _MATCH_ALL} if _MATCH_ALL is not None else {}
    tokens = _kiwi.tokenize(text, **kwargs)
    result = []
    for t in tokens:
        start = t.start
        end = getattr(t, "end", None)
        if end is None:
            end = start + t.len
        result.append({"word": t.form, "pos": t.tag, "start": start, "end": end})
    return result


def _char_shape(token: str) -> str:
    out = []
    for ch in token:
        if "\uac00" <= ch <= "\ud7a3" or "\u3131" <= ch <= "\u318e":
            out.append("H")
        elif ch.isdigit():
            out.append("D")
        elif ch.isalpha():
            out.append("L")
        else:
            out.append("S")
    return "".join(out)


def _has_digit(token: str) -> bool:
    return any(ch.isdigit() for ch in token)


def _has_special(token: str) -> bool:
    return any(not (ch.isalnum() or "\uac00" <= ch <= "\ud7a3") for ch in token)


# --- 2. feature dict 생성 (학습 스키마와 동일) ---
def build_token_features(text: str) -> List[Dict[str, Any]]:
    """반환값 각 원소에 '_span' 키로 (start,end) 를 같이 담는다.
    dict_vectorizer.transform 에 넘기기 전에 이 키는 제외해야 함
    (아래 process_message 에서 자동으로 처리)."""
    tokens = tokenize_with_span(text)
    words = [t["word"] for t in tokens]
    n = len(tokens)

    features = []
    for i in range(n):
        cur = words[i]
        feat = {
            "current_word": cur,
            "prev1_word": words[i - 1] if i - 1 >= 0 else "BOS1",
            "prev2_word": words[i - 2] if i - 2 >= 0 else "BOS2",
            "next1_word": words[i + 1] if i + 1 < n else "EOS1",
            "next2_word": words[i + 2] if i + 2 < n else "EOS2",
            "current_pos": tokens[i]["pos"],
            "current_shape": _char_shape(cur),
            "current_has_digit": int(_has_digit(cur)),
            "current_has_special": int(_has_special(cur)),
            "_span": (tokens[i]["start"], tokens[i]["end"]),
        }
        features.append(feat)
    return features


# --- 3. 메인 함수: Streamlit 팀이 이것 하나만 호출하면 됨 ---
def process_message(
    text: str,
    dict_vectorizer,
    runtime,
    threshold: float = DEFAULT_THRESHOLD,
    mode: Literal["block", "mask"] = "block",
    mask_char: str = "*",
) -> Dict[str, Any]:
    """
    Streamlit 팀이 호출할 단 하나의 함수.

    mode="block" (기본, 예시 1번 방식):
        민감정보가 하나라도 감지되면 메시지 전체를 막고, 안내 문구를 반환한다.
        {
          "action": "block",
          "display_message": "민감정보가 포함되어있습니다. 전송이 불가능합니다. 내용을 수정해 다시 전송해주세요.",
          "send_text": None,   # 전송 불가
        }

    mode="mask" (예시 2번 방식):
        민감정보 부분만 마스킹(****)하고, 마스킹된 텍스트를 그대로 전송 가능하게 한다.
        {
          "action": "mask",
          "display_message": "카드 번호는 ****************입니다.",
          "send_text": "카드 번호는 ****************입니다.",  # 이 값을 전송
        }

    둘 다 민감정보가 없으면:
        {
          "action": "pass",
          "display_message": text,
          "send_text": text,
        }
    """
    feats = build_token_features(text)
    if not feats:
        return {"action": "pass", "display_message": text, "send_text": text, "flagged_spans": []}

    spans = [f.pop("_span") for f in feats]
    X = dict_vectorizer.transform(feats)
    results = runtime.predict_from_features(X, threshold=threshold)

    flagged = []
    for (start, end), r in zip(spans, results):
        if r.blocked or r.warning:
            label = r.top_labels[0] if r.top_labels else None
            flagged.append({
                "start": start, "end": end, "label": label,
                "risk_score": r.risk_score, "blocked": r.blocked,
                "low_confidence": label in LOW_CONFIDENCE_LABELS,
            })

    if not flagged:
        return {"action": "pass", "display_message": text, "send_text": text, "flagged_spans": []}

    if mode == "block":
        return {
            "action": "block",
            "display_message": "민감정보가 포함되어있습니다. 전송이 불가능합니다. 내용을 수정해 다시 전송해주세요.",
            "send_text": None,
            "flagged_spans": flagged,
        }

    # mode == "mask"
    masked = text
    for f in sorted(flagged, key=lambda f: -f["start"]):
        s, e = f["start"], f["end"]
        masked = masked[:s] + mask_char * (e - s) + masked[e:]
    return {
        "action": "mask",
        "display_message": masked,
        "send_text": masked,
        "flagged_spans": flagged,
    }


if __name__ == "__main__":
    # 동작 확인 (실제 모델 연결 없이 토큰화 결과만 확인)
    sample = "카드번호는 0000-0000-0000-0000입니다."
    for f in build_token_features(sample):
        print(f)
