"""
  - 환각 방지 및 과적합 방어 로직
  - exact_target 필드 도입을 통한 핀셋 마스킹 (과잉 검열 방지)
  - 파이썬 후처리를 통한 글자 수 100% 보존
"""
import os
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv # [추가] .env 파일을 읽어오기 위한 라이브러리

from label_mapping import map_to_shieldchat_label
from llm_masking_extension import LABEL_SPEC_BY_NAME, _format_label_spec_for_prompt

# .env 파일 로드
load_dotenv()


# --- Pydantic Output 규격 (exact_target 추가) ---
class MLFindingReview(BaseModel):
    excerpt: str = Field(description="ML이 민감정보로 판단한 원문 조각 (덩어리)")
    original_label: str = Field(description="ML이 매긴 원래 라벨")
    verdict: Literal["confirm", "reject", "relabel"] = Field(
        description="confirm=ML 판단이 맞음, reject=ML이 틀림(오탐, 마스킹 안 함), relabel=라벨 변경"
    )
    corrected_label: Optional[str] = Field(
        default=None, description="verdict가 relabel인 경우에만: 올바른 라벨"
    )
    reason: str = Field(description="판단 이유")
    exact_target: str = Field(
        description="verdict가 confirm/relabel일 경우, excerpt 안에서 실제로 마스킹해야 할 '정확한 기밀 핵심 단어' (예: 계좌번호 숫자만, 전화번호만 좁혀서 추출). reject면 빈 문자열."
    )

class AdditionalFinding(BaseModel):
    excerpt: str = Field(description="새로 찾은 민감정보 조각 (덩어리)")
    label: str = Field(description="해당 라벨")
    reason: str = Field(description="판단 이유")
    exact_target: str = Field(description="실제로 마스킹해야 할 '정확한 기밀 핵심 단어'")

class ReviewResult(BaseModel):
    ml_reviews: List[MLFindingReview] = Field(description="ML 판단 검토 결과")
    additional_findings: List[AdditionalFinding] = Field(description="추가 발견 항목")
    final_masked_text: str = Field(description="최종 마스킹된 텍스트 (파이썬이 덮어쓸 임시 필드)")


# --- 시스템 프롬프트 (핀셋 추출 지시 추가) ---
SYSTEM_PROMPT_TEMPLATE = """당신은 사내 메신저 민감정보 검열 '2차 검토자(보안 엔지니어)'입니다.

1차로 머신러닝(ML) 모델과 정규식이 의심 구간을 찾았습니다. ML 모델은 종종 기밀 주변의 일상 단어까지 한 덩어리로 묶어서(예: "국민은행 123-456 예금주") 의심 구간으로 넘겨줍니다. 당신의 임무는 이를 정밀하게 필터링하고 핵심만 추출하는 것입니다.

=== 이미 확정된 것 (정규식이 찾은 것, 마스킹 대상) ===
{regex_confirmed}

=== ML 모델의 1차 판단 (당신의 꼼꼼한 검토가 필요함) ===
{ml_findings}

=== 판단 기준 (라벨 정의) ===
{label_definitions}

=== 당신이 준수해야 할 핵심 규칙 ===
1. [판단 및 정밀 추출 (핀셋 역할)]
   - ML의 판단이 맞다면(confirm/relabel), `exact_target` 필드에 덩어리 전체를 적지 말고 **반드시 실제 기밀에 해당하는 핵심 데이터(예: "1234-5678" 등 숫자나 아이디 부분)만 정확히 발췌**하여 적으세요.
   - 절대 주의: 일상 대화 문맥이더라도, 포함된 '실제 계좌번호', '실제 전화번호', '개인 ID' 등은 무조건 `confirm` 처리하고 핵심만 `exact_target`으로 추출하세요.

2. [문맥적 예외 허용 (Contextual Bypass) - reject 처리]
   문맥을 읽었을 때 다음 중 하나라면 유출이 아니므로 무조건 reject 하세요. (reject 시 exact_target은 비움)
   - ① 테스트/더미 데이터: "테스트용 계좌번호 123-456", "가짜 주민번호" 등.
   - ② 공용/공개 정보: '1588-1111', 'www.company.com' 등 사내외 공개 정보.
   - ③ 일반 명사 오탐: 회사명이 아닌 일반 단어로 쓰인 "기업", "청소" 등.

3. [이중 마스킹 방지]
   - 이미 `*` 처리가 된 데이터가 있다면 reject 처리하세요.
"""


class ReviewMasker:
    def __init__(self):
        # 환경 변수(.env)에서 API 키를 가져옵니다.
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 가 비어 있습니다. .env 파일을 확인하세요.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o" # settings.llm_model 대신 사용할 모델명 직접 입력

    def review_and_mask(
        self,
        text: str,
        ml_findings: List[dict],
        regex_findings: List[dict],
    ) -> ReviewResult:
        
        regex_lines = []
        for r in regex_findings:
            mapped = map_to_shieldchat_label(r["entity_type"])
            regex_lines.append(f'- "{r["matched_text"]}" ({mapped}, 원 유형: {r["entity_type"]})')
        regex_confirmed_str = "\n".join(regex_lines) if regex_lines else "(없음)"

        ml_lines = []
        all_labels_involved = set()
        for f in ml_findings:
            excerpt = text[f["start"]:f["end"]]
            ml_lines.append(f'- "{excerpt}" (라벨: {f["label"]})')
            all_labels_involved.add(f["label"])
        ml_findings_str = "\n".join(ml_lines) if ml_lines else "(없음)"

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
        
        result = completion.choices[0].message.parsed


        # 강제 후처리 로직 (LLM의 핀셋 추출 데이터 기반 마스킹)
        spans_to_mask = []
        
        # 정규식 확정분 추가
        for r in regex_findings:
            spans_to_mask.append(r["matched_text"])
            
        # LLM이 CONFIRM 한 결과 중 '정밀 추출된' 타겟만 추가
        for rev in result.ml_reviews:
            if rev.verdict in ["confirm", "relabel"]:
                # ML의 거친 excerpt 대신, LLM이 정밀하게 좁혀준 exact_target 사용
                target = rev.exact_target.strip(' \'"')
                
                # 타겟이 비어있거나 원문에 없는 엉뚱한 말이라면 패스
                if not target or target not in text:
                    continue
                
                # 이미 마스킹된 데이터 방어
                if '*' in target:
                    rev.verdict = "reject"
                    rev.reason = "시스템 강제 규칙: 이미 마스킹된 데이터 방지"
                # 공용 정보 예외
                elif target in ["1588-1111", "www.company.com"]:
                    rev.verdict = "reject"
                    rev.reason = "시스템 강제 규칙: 공용/공개 정보"
                else:
                    spans_to_mask.append(target)
        
        # 추가 발견분
        for add in result.additional_findings:
            target = add.exact_target.strip(' \'"')
            if target and target in text and '*' not in target:
                spans_to_mask.append(target)

        # 파이썬으로 정확한 길이 치환
        final_text = text
        spans_to_mask.sort(key=len, reverse=True)
        
        for span in spans_to_mask:
            if span in final_text:
                final_text = final_text.replace(span, '*' * len(span))
                
        result.final_masked_text = final_text

        return result