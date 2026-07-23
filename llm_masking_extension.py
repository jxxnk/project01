"""
llm_masking_extension.py
===========================
흐름:
  1. 우리 ML/정규식 모델이 이미 "이 메시지에 PHONE, AFFILIATION 라벨이 있다"
     까지는 판단해놓음 (hybrid_censor.py 의 결과).
  2. 그 라벨 이름들과, 각 라벨이 정확히 뭘 의미하는지에 대한 정의(ShieldChat
     스펙)를 GPT에게 같이 전달.
  3. GPT는 원문 텍스트 + 라벨 정의를 보고, "이 라벨에 해당하는 부분이 문장의
     어디인지" 직접 찾아서 마스킹된 텍스트를 반환.

이 방식의 장점: 우리 모델이 정확한 위치(span)를 몰라도(예: ML 모델은 라벨만
어느 정도 확신하고 정확한 글자 위치는 불확실), GPT가 그 라벨의 "형식"을 알고
문장에서 스스로 찾아내므로 span 계산 부담이 줄어듦.
"""
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from config_1 import settings


# --- 라벨 정의 (ShieldChat_합성데이터셋_1_T_.json 의 'labels' 필드 그대로) ---
LABEL_SPEC = [
    {
        "label": "NORMAL",
        "description": "민감정보나 개인정보가 없는 일반 대화",
        "entity_types": [],
        "decision_rule": "실제 민감정보 값이나 비밀정보가 포함되지 않은 경우",
    },
    {
        "label": "CONTACT_ID",
        "description": "직접 연락하거나 특정 계정이나 사용자를 식별할 수 있는 정보",
        "entity_types": ["PHONE", "EMAIL", "USER_ID", "SNS_ID"],
        "decision_rule": "전화번호, 이메일, 사용자 ID, SNS ID 등이 포함된 경우",
    },
    {
        "label": "FINANCIAL",
        "description": "금전이나 금융계정과 관련된 직접적인 민감정보",
        "entity_types": ["ACCOUNT_NUMBER", "CARD_NUMBER", "PAYMENT_INFO", "PAYMENT_AMOUNT"],
        "decision_rule": "계좌번호, 카드번호, 결제정보, 계약금액 등이 포함된 경우",
    },
    {
        "label": "PERSONAL_PROFILE",
        "description": "개인의 특성이나 신상과 관련된 정보",
        "entity_types": ["AGE", "SEX", "COUNTRY", "NICKNAME"],
        "decision_rule": "나이, 성별, 국적, 별명 등 개인의 속성정보가 포함된 경우",
    },
    {
        "label": "AFFILIATION",
        "description": "개인이 어느 조직에 속하는지, 어떤 직책이나 배경을 가졌는지 나타내는 정보",
        "entity_types": ["POSITION", "DEPARTMENT", "WORKPLACE", "EDUCATION", "MAJOR", "RELIGION", "CLUB"],
        "decision_rule": "직책, 부서, 직장, 학교, 전공, 종교, 동호회 등이 포함된 경우",
    },
    {
        "label": "LOCATION_ASSET",
        "description": "주소, 차량번호처럼 개인이나 장소, 자산을 직접 식별할 수 있는 정보",
        "entity_types": ["ADDRESS", "PLACE", "PLATE_NUMBER"],
        "decision_rule": "주소, 상세 장소, 차량번호 등이 포함된 경우",
    },
    {
        "label": "CREDENTIAL_SECRET",
        "description": "시스템 또는 계정 접근에 사용되는 인증 비밀정보",
        "entity_types": ["API_KEY", "PASSWORD", "ACCESS_TOKEN", "PRIVATE_KEY", "SERVER_ACCOUNT"],
        "decision_rule": "API 키, 비밀번호, 토큰, 개인키, 서버 계정 등이 포함된 경우",
    },
]
LABEL_SPEC_BY_NAME = {l["label"]: l for l in LABEL_SPEC}


def _format_label_spec_for_prompt(target_labels: List[str]) -> str:
    """프롬프트에 넣을 라벨 정의 텍스트를 만든다. 이번 메시지에서 실제로
    걸린 라벨들만 추려서 넣는다 (불필요하게 전체 7개를 다 보내지 않음)."""
    lines = []
    for name in target_labels:
        spec = LABEL_SPEC_BY_NAME.get(name)
        if not spec:
            continue
        lines.append(
            f"- {spec['label']}: {spec['description']}\n"
            f"  세부 유형: {', '.join(spec['entity_types']) or '(해당없음)'}\n"
            f"  판단 기준: {spec['decision_rule']}"
        )
    return "\n".join(lines)


# --- GPT 응답 구조 ---
class LabelGuidedMaskResult(BaseModel):
    masked_text: str = Field(description="라벨에 해당하는 부분을 '*'로 마스킹한 전체 문장")
    found_spans: List[str] = Field(description="실제로 마스킹한 원문 부분 텍스트 목록 (마스킹 전 값)")


SYSTEM_PROMPT_TEMPLATE = """당신은 사내 메신저 민감정보 마스킹 엔진입니다.

이미 1차 분석에서 아래 라벨들이 이 문장에 포함된 것으로 판단되었습니다.
당신의 역할은 문장을 직접 읽고, 각 라벨의 정의에 해당하는 "구체적인 부분"을
찾아서 그 부분만 '*' 문자로 마스킹하는 것입니다.

=== 이번 문장에서 감지된 라벨 정의 ===
{label_definitions}

=== 작업 규칙 ===
1. 위 라벨 정의의 '세부 유형'과 '판단 기준'에 해당하는 구체적인 텍스트 조각만
   찾아서 마스킹하세요 (예: PHONE 라벨이면 전화번호 형식의 숫자만, 문장 전체를
   마스킹하면 안 됩니다).
2. 마스킹한 길이만큼 '*'로 치환하세요 (예: "010-1234-5678" -> "*************").
3. 라벨과 무관한 일반적인 단어(조사, 흔한 명사 등)는 절대 마스킹하지 마세요.
4. 확신이 서지 않는 부분은 마스킹하지 말고 그대로 두세요 (과잉 마스킹 금지).
"""


class LabelGuidedMasker:
    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY 가 비어 있습니다. .env 파일을 확인하세요.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model

    def mask(self, text: str, detected_labels: List[str]) -> LabelGuidedMaskResult:
        """
        text: 원문 메시지
        detected_labels: 우리 모델(hybrid_censor.py 등)이 이미 판단한 라벨 이름 리스트
                         예: ["PHONE적용라벨" 대신 실제로는 ["CONTACT_ID", "AFFILIATION"] 같은 대표라벨]
        """
        if not detected_labels:
            return LabelGuidedMaskResult(masked_text=text, found_spans=[])

        label_defs = _format_label_spec_for_prompt(detected_labels)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(label_definitions=label_defs)

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"다음 문장을 마스킹하세요:\n\n{text}"},
            ],
            response_format=LabelGuidedMaskResult,
        )
        return completion.choices[0].message.parsed


if __name__ == "__main__":
    # 사용 예시: 우리 모델(hybrid_censor.py)이 이미 라벨을 판단해서 넘겨준 상황을 가정
    masker = LabelGuidedMasker()

    # 예시 1: 우리 정규식이 CONTACT_ID(전화번호)를 감지했다고 가정
    result = masker.mask(
        "제 번호는 010-1234-5678이고 이메일은 test@example.com이에요.",
        detected_labels=["CONTACT_ID"],
    )
    print("마스킹 결과:", result.masked_text)
    print("찾은 부분:", result.found_spans)

    # 예시 2: AFFILIATION(소속) 라벨이 감지된 경우
    result2 = masker.mask(
        "저는 개발팀 소속 대리이고, 서울대학교 출신입니다.",
        detected_labels=["AFFILIATION"],
    )
    print("\n마스킹 결과:", result2.masked_text)
    print("찾은 부분:", result2.found_spans)
