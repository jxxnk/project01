"""
label_mapping.py
==================
pii_masker.py 가 쓰는 세부 라벨(형식 기반) <-> 대표 라벨(ShieldChat 체계)
매핑표.
"""

PII_MASKER_TO_SHIELDCHAT_LABEL = {
    "PHONE": "CONTACT_ID",
    "EMAIL": "CONTACT_ID",
    "SNS_ID": "CONTACT_ID",
    "CARD_NUMBER": "FINANCIAL",
    "CARD_NUMBER_NO_SEP": "FINANCIAL",
    "ACCOUNT_NUMBER": "FINANCIAL",
    "API_KEY_OR_TOKEN": "CREDENTIAL_SECRET",
    "PLATE_NUMBER": "LOCATION_ASSET",
    "RESIDENT_REGISTRATION_NUMBER": "CREDENTIAL_SECRET",
}

def map_to_shieldchat_label(pii_masker_entity_type: str) -> str:
    """매핑표에 없는 타입이 들어오면 원래 이름을 그대로 반환"""
    return PII_MASKER_TO_SHIELDCHAT_LABEL.get(pii_masker_entity_type, pii_masker_entity_type)
