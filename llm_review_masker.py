"""
label_mapping.py
==================
pii_masker.py 가 쓰는 세부 라벨(형식 기반) <-> 대표 라벨(ShieldChat 체계)
매핑표.

process_message()(ML/Kiwi 모델)는 이미 7개 대표 라벨(AFFILIATION, CONTACT_ID,
CREDENTIAL_SECRET, FINANCIAL, LOCATION_ASSET, NORMAL, PERSONAL_PROFILE)로
결과를 냄. 반면 pii_masker.py 는 형식 기반으로 더 세분화된 이름(PHONE,
CARD_NUMBER 등)을 씀. LLM 프롬프트에는 한 가지 체계로 통일해서 넘겨야
라벨 정의(LABEL_SPEC)를 조회할 수 있으므로, 이 매핑표로 변환한다.
"""

# pii_masker.py의 entity_type -> 7개 대표 라벨(ShieldChat 체계)
PII_MASKER_TO_SHIELDCHAT_LABEL = {
    # ShieldChat 원본 스펙 기준 (entity_types 필드 참고):
    #   CONTACT_ID: PHONE, EMAIL, USER_ID, SNS_ID
    #   FINANCIAL: ACCOUNT_NUMBER, CARD_NUMBER, PAYMENT_INFO, PAYMENT_AMOUNT
    #   CREDENTIAL_SECRET: API_KEY, PASSWORD, ACCESS_TOKEN, PRIVATE_KEY, SERVER_ACCOUNT
    #   LOCATION_ASSET: ADDRESS, PLACE, PLATE_NUMBER
    "PHONE": "CONTACT_ID",
    "EMAIL": "CONTACT_ID",
    "SNS_ID": "CONTACT_ID",

    "CARD_NUMBER": "FINANCIAL",
    "CARD_NUMBER_NO_SEP": "FINANCIAL",
    "ACCOUNT_NUMBER": "FINANCIAL",

    "API_KEY_OR_TOKEN": "CREDENTIAL_SECRET",

    "PLATE_NUMBER": "LOCATION_ASSET",

    # [주의] 주민등록번호(RESIDENT_REGISTRATION_NUMBER)는
    # 카테고리 정의 어디에도 정확히 안 맞음 -- 실무 요청으로 pii_masker.py에
    # 추가한 항목이라 원본 스펙엔 없음. 가장 가까운 것으로 CREDENTIAL_SECRET
    # (신원 확인/인증에 준하는 용도로 쓰임)에 매핑했지만, 이건 판단이 갈릴 수
    # 있는 부분이라 팀 내에서 확정 필요. 필요하면 새 카테고리
    # "REGISTRATION_NUMBER"를 만들어 7개 체계에 추가하는 것도 고려.
    "RESIDENT_REGISTRATION_NUMBER": "CREDENTIAL_SECRET",
}


def map_to_shieldchat_label(pii_masker_entity_type: str) -> str:
    """매핑표에 없는 타입이 들어오면 원래 이름을 그대로 반환 (누락 방지용
    안전장치 -- 매핑표를 계속 업데이트해야 한다는 신호이기도 함)."""
    return PII_MASKER_TO_SHIELDCHAT_LABEL.get(pii_masker_entity_type, pii_masker_entity_type)
