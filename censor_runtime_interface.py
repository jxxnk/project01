#1. Imports
import io
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


#2. Response schema
@dataclass
class CensorResult:
    blocked: bool
    warning: bool
    top_labels: List[str]
    risk_score: float
    threshold: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


#3. Core runtime
class CensorRuntime:
    """
    실시간 메시지/첨부파일 검열용 런타임 인터페이스.

    이 클래스는 raw text 를 바로 받는 모델(final_raw_text_model.joblib) 과
    함께 쓰도록 되어 있다. model.predict_proba(list_of_raw_texts) 를 호출하므로
    dict_vectorizer/형태소 분석기 같은 별도 전처리기가 필요 없다.
    """

    def __init__(self, model_bundle_path: str, class_names: Optional[List[str]] = None):
        bundle = joblib.load(model_bundle_path)
        self.model = bundle["model"]
        self.class_names = class_names or bundle.get("class_names", [])
        self.raw_problem_type = bundle.get("raw_problem_type", "unknown")
        self.inference_mode = bundle.get("inference_mode", "unknown")
        # '정상' 클래스는 위험도 계산에서 반드시 제외해야 한다 (아래 참고)
        normal_name = bundle.get("normal_class", "NORMAL")
        self.normal_idx = (
            self.class_names.index(normal_name) if normal_name in self.class_names else None
        )

    def predict_from_features(self, X_features, threshold: float = 0.5) -> List[CensorResult]:
        """
        X_features: List[str] (raw text 리스트). final_raw_text_model.joblib 은
        내부적으로 벡터화까지 다 처리하므로 그대로 문자열 리스트를 넘기면 된다.
        """
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_features)
            if isinstance(proba, list):
                score_matrix = np.column_stack([p[:, 1] for p in proba])
            else:
                score_matrix = np.asarray(proba)
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X_features)
            score_matrix = 1.0 / (1.0 + np.exp(-np.asarray(scores)))
        else:
            pred = self.model.predict(X_features)
            score_matrix = np.asarray(pred, dtype=float)

        if score_matrix.ndim == 1:
            score_matrix = score_matrix.reshape(-1, 1)

        results: List[CensorResult] = []
        for row in score_matrix:
            # risk_score/순위는 반드시 '민감정보 클래스들'만 대상으로 계산한다.
            # NORMAL까지 포함해서 최댓값을 구하면, 모델이 "정상"이라고 강하게
            # 확신할수록 risk_score 도 같이 높아져서 명백한 정상 메시지까지
            # 차단되는 오류가 생기기 때문이다.
            if self.normal_idx is not None and len(row) > self.normal_idx:
                candidate_idx = [i for i in range(len(row)) if i != self.normal_idx]
            else:
                candidate_idx = list(range(len(row)))

            sorted_candidates = sorted(candidate_idx, key=lambda i: row[i], reverse=True)
            top_idx = sorted_candidates[: min(3, len(sorted_candidates))]
            top_labels = [self.class_names[i] if i < len(self.class_names) else str(i) for i in top_idx]
            risk_score = float(row[top_idx[0]]) if top_idx else 0.0
            blocked = risk_score >= threshold
            warning = (risk_score >= threshold * 0.7) and not blocked
            reason = (
                "민감정보 유출 가능성이 높은 패턴 감지"
                if blocked
                else "경고 임계치 인접"
                if warning
                else "허용 임계치 이하"
            )
            results.append(
                CensorResult(
                    blocked=blocked,
                    warning=warning,
                    top_labels=top_labels,
                    risk_score=risk_score,
                    threshold=threshold,
                    reason=reason,
                )
            )
        return results


#4. File-to-text helpers
class AttachmentTextExtractor:
    """첨부파일 -> 텍스트 추출. TXT/CSV/PDF 지원."""

    @staticmethod
    def from_txt(file_bytes: bytes, encoding: str = "utf-8") -> str:
        return file_bytes.decode(encoding, errors="ignore")

    @staticmethod
    def from_csv(file_bytes: bytes) -> str:
        df = pd.read_csv(io.BytesIO(file_bytes))
        return "\n".join(df.astype(str).fillna("").agg(" ".join, axis=1).tolist())

    @staticmethod
    def from_pdf(file_bytes: bytes) -> str:
        """pypdf 로 페이지별 텍스트를 추출해 합친다.
        스캔본(이미지) PDF는 텍스트가 없어 빈 문자열이 나올 수 있음 -- 이 경우
        OCR이 별도로 필요하며, 이 함수 범위 밖이다."""
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
        return "\n".join(pages_text)


#5. Integration notes
INTEGRATION_CHECKLIST = {
    "streamlit_send_flow": [
        "사용자 메시지 입력",
        "첨부파일이 있으면 AttachmentTextExtractor 로 텍스트 추출",
        "메시지 본문 + 첨부 텍스트 결합 또는 개별 평가",
        "runtime.predict_from_features([텍스트], threshold) 호출",
        "blocked/warning 결과를 Streamlit UI 와 GPT 호출 전에 반영",
    ],
    "hard_block_example": {
        "blocked": True,
        "warning": False,
        "top_labels": ["FINANCIAL", "CONTACT_ID"],
        "risk_score": 0.94,
        "threshold": 0.5,
        "reason": "민감정보 유출 가능성이 높은 패턴 감지",
    },
    "warning_example": {
        "blocked": False,
        "warning": True,
        "top_labels": ["PERSONAL_PROFILE"],
        "risk_score": 0.4,
        "threshold": 0.5,
        "reason": "경고 임계치 인접",
    },
}
