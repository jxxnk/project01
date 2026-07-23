import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Settings:
    def __init__(self):
        # .env 파일에서 OPENAI_API_KEY를 자동으로 가져옵니다.
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # 팀원 코드가 사용할 기본 모델 이름 설정 (gpt-4o 등으로 변경 가능)
        self.llm_model = "gpt-5.5"

# llm_censor_3.py 등에서 settings 객체를 임포트할 수 있도록 인스턴스화
settings = Settings()