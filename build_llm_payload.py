"""
llm_review_masker.py
=======================
요청사항 반영:
  "처음에는 ML을 우선하되, GPT가 ML이 틀렸다고 판단하면 수정할 수 있게."

흐름:
  1. 정규식(pii_masker.py) 결과 -> 항상 신뢰(REGEX_CONFIRMED), GPT에게도
     "이건 이미 확정된 것"으로 전달 (검토 대상 아님).
  2. ML/Kiwi 토큰 모델(process_message()) 결과 -> "1차 판단(안)"으로 GPT에게
     전달. 각 판단에 원문 조각 + 라벨 + 확신도(risk_score) + low_confidence
     여부를 같이 줌.
  3. GPT가 각 ML 판단에 대해:
       - confirm  : ML 판단이 맞다고 봄 -> 그대로 마스킹
       - reject   : ML이 틀렸다고 봄(오탐) -> 마스킹 안 함
       - relabel  : 라벨은 틀렸지만 민감정보는 맞다고 봄 -> 라벨만 교체
     그리고 ML/정규식이 놓친 것으로 보이는 부분이 있으면 추가로 찾아냄.
  4. 최종적으로 (정규식 확정 + GPT가 confirm/relabel한 ML 판단 + GPT가 새로
     찾은 것)을 종합해 마스킹된 텍스트를 만듦.

이 방식의 핵심: ML이 "기업", "청소" 같은 흔한 단어에 오탐했던 사례들을,
GPT가 문맥을 보고 "이건 민감정보 아님(reject)"으로 걸러낼 수 있음.
"""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

from config_1 import settings
from label_mapping import map_to_shieldchat_label

# 라벨 정의 (llm_masking_extension.py 와 동일 -- ShieldChat 원본 스펙)
from llm_masking_extension import LABEL_SPEC_BY_NAME, _format_label_spec_for_prompt


class MLFindingReview(BaseModel):
    excerpt: str = Field(description="ML이 민감정보로 판단한 원문 조각")
    original_label: str = Field(description="ML이 매긴 원래 라벨")
    verdict: Literal["confirm", "reject", "relabel"] = Field(
        description="confirm=ML 판단이 맞음, reject=ML이 틀림(오탐, 마스킹 안 함), "
                    "relabel=민감정보는 맞으나 라벨이 틀림"
    )
    corrected_label: Optional[str] = Field(
        default=None, description="verdict가 relabel인 경우에만: 올바른 라벨"
    )
    reason: str = Field(description="이렇게 판단한 이유 (간단히)")


class AdditionalFinding(BaseModel):
    excerpt: str = Field(description="ML/정규식이 놓쳤지만 GPT가 새로 찾은 민감정보 조각")
    label: str = Field(description="해당 라벨")
    reason: str = Field(description="왜 민감정보라고 판단했는지")


class ReviewResult(BaseModel):
    ml_reviews: List[MLFindingReview] = Field(description="ML 판단 각각에 대한 검토 결과")
    additional_findings: List[AdditionalFinding] = Field(
        description="ML/정규식이 놓친 것으로 보이는 추가 발견 항목 (없으면 빈 리스트)"
    )
    final_masked_text: str = Field(
        description="최종적으로 확정된 모든 민감정보 부분을 '*'로 마스킹한 전체 문장"
    )


SYSTEM_PROMPT_TEMPLATE = """당신은 사내 메신저 민감정보 검열의 '2차 검토자'입니다.

1차로 머신러닝 모델이 이미 아래와 같이 판단해두었습니다. 이 판단은 참고용
'1차 의견'이며, 당신이 이걸 그대로 믿을 필요는 없습니다 -- 문맥을 보고
직접 판단하세요.

=== 이미 확정된 것 (정규식이 찾은 것, 검토 대상 아님, 그대로 마스킹) ===
{regex_confirmed}

=== ML 모델의 1차 판단 (검토 필요) ===
{ml_findings}

=== 라벨 정의 ===
{label_definitions}

=== 당신이 할 일 ===
1. ML의 1차 판단 각각에 대해 confirm(맞음) / reject(틀림, 오탐) /
   relabel(민감정보는 맞으나 라벨이 틀림) 중 하나로 판정하세요.
   - 흔한 일상 단어(예: 회사명이 아닌 일반명사로 쓰인 "기업", 직업과 무관한
     "교수" 언급, 일상적인 "청소" 이야기 등)가 실제 맥락과 무관하게 걸린
     경우는 반드시 reject 하세요.
2. ML과 정규식이 놓쳤지만 명백히 민감정보로 보이는 부분이 있으면
   additional_findings 에 추가하세요.
3. 최종적으로 [정규식 확정분 + confirm/relabel된 ML 판단 + 추가 발견]을
   전부 반영해서, 해당 부분만 '*'로 마스킹한 전체 문장을 final_masked_text
   에 작성하세요. 나머지 문장은 원문 그대로 유지하세요.
"""


class ReviewMasker:
    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY 가 비어 있습니다. .env 파일을 확인하세요.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model

    def review_and_mask(
        self,
        text: str,
        ml_findings: List[dict],      # [{"start","end","label","risk_score","low_confidence"}, ...]
        regex_findings: List[dict],   # [{"start","end","entity_type","matched_text"}, ...]
    ) -> ReviewResult:
        # 정규식 확정분 (검토 대상 아님, ShieldChat 7개 라벨로 매핑해서 표시)
        regex_lines = []
        for r in regex_findings:
            mapped = map_to_shieldchat_label(r["entity_type"])
            regex_lines.append(f'- "{r["matched_text"]}" ({mapped}, 원 유형: {r["entity_type"]}) -- 확정, 무조건 마스킹')
        regex_confirmed_str = "\n".join(regex_lines) if regex_lines else "(없음)"

        # ML 1차 판단 (검토 대상)
        ml_lines = []
        all_labels_involved = set()
        for f in ml_findings:
            excerpt = text[f["start"]:f["end"]]
            confidence_note = " [신뢰도 낮음, 신중히 검토]" if f.get("low_confidence") else ""
            ml_lines.append(
                f'- "{excerpt}" (라벨: {f["label"]}, 확신도: {f["risk_score"]:.2f}){confidence_note}'
            )
            all_labels_involved.add(f["label"])
        ml_findings_str = "\n".join(ml_lines) if ml_lines else "(없음)"

        # 라벨 정의는 이번 판단에 관련된 것만 (정규식쪽 매핑 라벨도 포함)
        for r in regex_findings:
            all_labels_involved.add(map_to_shieldchat_label(r["entity_type"]))
        label_defs_str = _format_label_spec_for_prompt(sorted(all_labels_involved)) or "(해당 없음)"

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            regex_confirmed=regex_confirmed_str,
            ml_findings=ml_findings_str,
            label_definitions=label_defs_str,
        )

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"검토할 원문 전체:\n\n{text}"},
            ],
            response_format=ReviewResult,
        )
        return completion.choices[0].message.parsed


if __name__ == "__main__":
    reviewer = ReviewMasker()

    # 예시: ML이 "기업"을 AFFILIATION으로 오탐한 상황을 가정
    text = "현대자동차는 역시 좋은 기업인 것 같아. 제 번호는 010-1234-5678이에요."
    ml_findings = [
        {"start": 8, "end": 10, "label": "AFFILIATION", "risk_score": 0.74, "low_confidence": False},
    ]
    regex_findings = [
        {"start": 34, "end": 47, "entity_type": "PHONE", "matched_text": "010-1234-5678"},
    ]

    result = reviewer.review_and_mask(text, ml_findings, regex_findings)
    print("=== ML 판단 검토 결과 ===")
    for r in result.ml_reviews:
        print(f"  {r.excerpt!r} ({r.original_label}) -> {r.verdict} ({r.reason})")
    print("\n=== 추가 발견 ===")
    for a in result.additional_findings:
        print(f"  {a.excerpt!r} ({a.label}) -- {a.reason}")
    print("\n=== 최종 마스킹 결과 ===")
    print(result.final_masked_text)
