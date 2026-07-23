"""
Streamlit 프론트엔드 연동을 위한 Adapter 모듈.
"""
from llm_censor_3 import ReviewMasker

def process_message_with_llm(original_text: str, ml_findings: list, regex_findings: list, mode: str = "mask") -> dict:
    # 1. LLM + 정규식 마스킹 파이프라인 실행
    reviewer = ReviewMasker()
    result = reviewer.review_and_mask(original_text, ml_findings, regex_findings)
    
    # 2. 원문과 마스킹된 결과를 비교하여 기밀 유무 판단
    is_masked = original_text != result.final_masked_text
    
    # 3. 로그 기록용 세부 내역 (flagged_spans) 정리
    audit_logs = []
    
    # 정규식으로 찾은 것 기록
    for r in regex_findings:
        audit_logs.append({
            "target": r["matched_text"], 
            "reason": f"정규식 확정 ({r['entity_type']})"
        })
        
    # LLM이 컨펌하거나 라벨을 변경한 것 기록
    for rev in result.ml_reviews:
        if rev.verdict in ["confirm", "relabel"]:
            audit_logs.append({
                "target": rev.exact_target, 
                "reason": f"LLM 검토 완료: {rev.reason}"
            })
            
    # 4. 프론트엔드 요구 규격(가이드라인)에 맞춰 딕셔너리 생성 및 반환
    if not is_masked:
        # 기밀이 없거나, LLM이 모두 예외(Reject) 처리한 경우 -> 정상 통과
        return {
            "action": "pass",
            "display_message": "",
            "send_text": original_text,
            "flagged_spans": audit_logs
        }
        
    else:
        # 기밀이 발견되어 마스킹 처리가 일어난 경우
        if mode == "block":
            return {
                "action": "block",
                "display_message": "민감정보가 포함되어있습니다. 전송이 불가능합니다. 내용을 수정해 다시 전송해주세요.",
                "send_text": None,
                "flagged_spans": audit_logs
            }
        else: # mode == "mask"
            return {
                "action": "mask",
                "display_message": "보안 정책에 따라 민감정보가 마스킹 처리되어 전송되었습니다.",
                "send_text": result.final_masked_text,
                "flagged_spans": audit_logs
            }