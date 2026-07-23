"""
pii_masker.py
==============
메시지 단위 분류 모델(final_raw_text_model.joblib)은 "이 문장에 민감정보가
있는지 없는지"만 판단하고, 어느 글자 범위(span)가 민감정보인지는 모른다.

이 모듈은 그 부분을 보완한다 -- 형식이 고정된(정규식으로 판별 가능한) 민감정보
유형에 한해, 문장 내 정확한 위치를 찾아 마스킹(********)까지 해준다.

⚠️ 커버 범위: 아래 정규식 목록에 있는, "형식이 고정된" 유형만 마스킹 가능.
닉네임/직책/부서/주소처럼 문맥에 따라 형태가 달라지는 자유 텍스트 유형은
정규식으로 잡을 수 없고, 별도의 위치 인식(NER) 모델이 있어야 한다.
(ShieldChat 원본 데이터의 entities 필드에 span 정답이 있으니, 이걸로 NER
모델을 학습하는 것이 다음 단계로 필요함 -- 이 파일은 그 전 단계의 실용적 대안)
"""
import re
from dataclasses import dataclass
from typing import List


@dataclass
class MaskSpan:
    start: int
    end: int
    matched_text: str
    entity_type: str


# (entity_type, 정규식, 우선순위) -- 우선순위 낮을수록 먼저 매칭 (겹칠 때 우선)
_PATTERNS = [
    # 주민등록번호: 6자리-7자리 (하이픈 있음/없음 둘 다) -- 카테고리 정의에는 없지만
    # 실무에서 요청하신 대로 새로 추가함. 필요 시 REGISTRATION_NUMBER 라는
    # 새 라벨로 7개 카테고리에 등록하는 것을 권장함 (아래 설명 참고)
    # [수정] \b 대신 (?<!\d)/(?!\d) lookaround 사용. 한글 음절도 정규식에서는
    # '단어 문자'(\w)로 취급되기 때문에, 숫자 바로 뒤에 한글이 붙어있으면
    # (예: "0000-0000-0000-0000입니다") \b가 경계로 인식을 안 해서 매칭이
    # 실패하는 버그가 있었음. (?!\d) 는 "다음이 숫자만 아니면 됨" 이라서
    # 한글이 바로 붙어도 정상적으로 매칭됨.
    ("RESIDENT_REGISTRATION_NUMBER", re.compile(r"(?<!\d)\d{6}[-\s]?\d{7}(?!\d)"), 0),

    # 카드번호: 4-4-4-4 형태 (구분자 하이픈/공백)
    ("CARD_NUMBER", re.compile(r"(?<!\d)\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}(?!\d)"), 1),

    # 전화번호: 010-1234-5678 형태 (지역번호 포함 변형도 커버)
    ("PHONE", re.compile(r"(?<!\d)0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}(?!\d)"), 2),

    # 계좌번호: 3-2~6-6~7 정도 형태로 넉넉하게 (은행마다 자릿수 다름)
    ("ACCOUNT_NUMBER", re.compile(r"(?<!\d)\d{2,6}-\d{2,6}-\d{4,8}(?!\d)"), 3),

    # 차량번호: '가상 12가 3456' 같은 국내 번호판 형식
    ("PLATE_NUMBER", re.compile(r"[가-힣]{0,3}\s?\d{2,3}[가-힣]\s?\d{4}(?!\d)"), 4),

    # 이메일
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), 5),

    # API 키 / 토큰 계열 (sk_, tok_, pk_ 등 흔한 접두사 + 영숫자)
    ("API_KEY_OR_TOKEN", re.compile(r"(?<![A-Za-z0-9_])(?:sk|pk|tok|api|key)[_-][A-Za-z0-9_]{6,}(?![A-Za-z0-9_])", re.IGNORECASE), 6),

    # SNS/사용자 ID (@로 시작)
    ("SNS_ID", re.compile(r"@[A-Za-z0-9_]{3,}"), 7),
]


def find_sensitive_spans(text: str) -> List[MaskSpan]:
    """정규식으로 찾을 수 있는 민감정보 span 목록을 반환한다.
    겹치는 매칭은 우선순위가 높은(숫자가 작은) 것을 남긴다."""
    candidates = []
    for entity_type, pattern, priority in _PATTERNS:
        for m in pattern.finditer(text):
            candidates.append((priority, m.start(), m.end(), m.group(), entity_type))

    # 겹치는 span 정리: 우선순위 -> 더 긴 매칭 순으로 정렬 후, 이미 마스킹된 범위와
    # 겹치면 제외
    candidates.sort(key=lambda c: (c[0], -(c[2] - c[1])))
    taken = []
    spans = []
    for priority, start, end, matched, etype in candidates:
        if any(not (end <= s or start >= e) for s, e in taken):
            continue  # 이미 다른 span과 겹침 -> 스킵
        taken.append((start, end))
        spans.append(MaskSpan(start=start, end=end, matched_text=matched, entity_type=etype))

    spans.sort(key=lambda s: s.start)
    return spans


def mask_text(text: str, mask_char: str = "*") -> str:
    """찾은 span을 전부 mask_char로 치환한 문자열을 반환한다."""
    spans = find_sensitive_spans(text)
    if not spans:
        return text
    out = []
    last = 0
    for s in spans:
        out.append(text[last:s.start])
        out.append(mask_char * (s.end - s.start))
        last = s.end
    out.append(text[last:])
    return "".join(out)


if __name__ == "__main__":
    examples = [
        "주민등록번호는 000000-0000000입니다",
        "주민등록번호는 0000000000000입니다",
        "카드번호는 0000-0000-0000-0000입니다.",
        "제 번호는 010-1234-5678입니다.",
        "계좌 000-12345-678901 로 송금했습니다.",
        "문의는 user1234@example.com 로 하세요.",
        "새 키는 sk_test_fake_12345678 입니다.",
        "오늘 점심 메뉴는 뭔가요?",  # 정상 문장 -- 마스킹 없어야 함
    ]
    for ex in examples:
        spans = find_sensitive_spans(ex)
        print(f"원문 : {ex}")
        print(f"마스킹: {mask_text(ex)}")
        print(f"찾은 span: {[(s.matched_text, s.entity_type) for s in spans]}")
        print()
