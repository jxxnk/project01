from llm_censor_3 import ReviewMasker

# --- 인덱스 자동 계산 도우미 함수 (하드코딩 버그 완벽 해결) ---
def mock_ml(text, word, label, risk_score, low_confidence=False):
    start = text.find(word)
    if start == -1:
        raise ValueError(f"오류: '{word}' 단어를 원문에서 찾을 수 없습니다.")
    return {"start": start, "end": start + len(word), "label": label, "risk_score": risk_score, "low_confidence": low_confidence}

def mock_regex(text, word, entity_type):
    start = text.find(word)
    if start == -1:
        raise ValueError(f"오류: '{word}' 단어를 원문에서 찾을 수 없습니다.")
    return {"start": start, "end": start + len(word), "entity_type": entity_type, "matched_text": word}

# --- 테스트 실행부 ---
def run_self_test():
    reviewer = ReviewMasker()
    
    # [테스트 케이스 모음]
    test_cases = [
        (
            "1. 기본기 (FINANCIAL)", 
            "c",
            [mock_ml("어제 회식비 정산 부탁드립니다. 국민은행 123401-04-567890 예금주 홍길동입니다.", "123401-04-567890", "FINANCIAL", 0.95)],
            []
        ),
        (
            "2. 오탐 걸러내기 (Reject 테스트 - 단어 자동 추출)",
            "우리는 글로벌 최고 기업으로 성장해야 합니다. 사무실 청소 상태도 신경 써주세요.",
            [
                mock_ml("우리는 글로벌 최고 기업으로 성장해야 합니다. 사무실 청소 상태도 신경 써주세요.", "기업", "AFFILIATION", 0.70, True),
                mock_ml("우리는 글로벌 최고 기업으로 성장해야 합니다. 사무실 청소 상태도 신경 써주세요.", "청소", "NORMAL", 0.60, True)
            ],
            []
        ),
        (
            "3. 공용 정보 예외 (파이썬 후처리 방어)",
            "제품에 문제가 생기면 고객센터 1588-1111 로 전화하거나 홈페이지 www.company.com 에 문의하세요.",
            [
                mock_ml("제품에 문제가 생기면 고객센터 1588-1111 로 전화하거나 홈페이지 www.company.com 에 문의하세요.", "1588-1111", "CONTACT_ID", 0.88),
                mock_ml("제품에 문제가 생기면 고객센터 1588-1111 로 전화하거나 홈페이지 www.company.com 에 문의하세요.", "www.company.com", "CONTACT_ID", 0.85)
            ],
            []
        ),
        (
            "4. 이중 마스킹 방어 (환각 테스트)",
            "고객님의 연락처 010-****-5678로 안내 문자를 발송했습니다.",
            [mock_ml("고객님의 연락처 010-****-5678로 안내 문자를 발송했습니다.", "010-****-5678", "CONTACT_ID", 0.90)],
            []
        ),
        (
            "5. 정규식 우선순위 및 혼합 테스트",
            "내 아이디는 developer_kim 이고, 비밀번호는 P@ssw0rd2026! 야.",
            [mock_ml("내 아이디는 developer_kim 이고, 비밀번호는 P@ssw0rd2026! 야.", "developer_kim", "CONTACT_ID", 0.85)], 
            [mock_regex("내 아이디는 developer_kim 이고, 비밀번호는 P@ssw0rd2026! 야.", "P@ssw0rd2026!", "PASSWORD")]
        ),
        (
            "6. [NEW] 문맥적 예외 허용 (프론트엔드 요청 기능)",
            "QA 부서입니다. 회원가입 테스트용 가짜 계좌번호 111-222-333333 으로 결제 연동 확인 부탁드립니다.",
            [mock_ml("QA 부서입니다. 회원가입 테스트용 가짜 계좌번호 111-222-333333 으로 결제 연동 확인 부탁드립니다.", "111-222-333333", "FINANCIAL", 0.99)],
            [] # 실제 계좌번호 형태라 ML은 무조건 위험하다고 판단했다고 가정
        )
    ]

    print("LLM 보안 필터 및 방어 로직 모의 테스트 (v1.0)\n" + "="*70)

    for i, (test_name, text, ml_mock, regex_mock) in enumerate(test_cases, 1):
        print(f"\n[TEST {i}] {test_name}")
        print(f"  입력 원문: {text}")
        
        try:
            result = reviewer.review_and_mask(text, ml_mock, regex_mock)
            
            if result.ml_reviews:
                for rev in result.ml_reviews:
                    # 콘솔 출력을 깔끔하게 다듬음
                    clean_verdict = f"[{rev.verdict.upper():^7}]" 
                    print(f"  ├─ ML 검토: '{rev.excerpt}' -> {clean_verdict} (사유: {rev.reason})")
            else:
                print("  ├─ ML 검토: (1차 의심 구간 없음)")
                
            print(f"  └─ 최종 출력: {result.final_masked_text}")
            
        except Exception as e:
            print(f"에러 발생: {e}")
            
    print("\n" + "="*70 + "\n테스트 완료!")

if __name__ == "__main__":
    run_self_test()