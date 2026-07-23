"""
민감정보 검열 앙상블 모델
=========================
두 개의 서로 다른 서브모델을 결합해 '위험 점수(risk score)'를 계산한다.

- ml_model  : OneVsRestClassifier(SGDClassifier) -> DictVectorizer가 만든
              희소 feature(X, shape=(N, 79811))에 직접 적용
- dl_model  : TruncatedSVD(64차원 축소) + MLPClassifier -> 위 X를 SVD로
              축소한 뒤 적용

두 모델의 클래스별 확률을 "요소별 최댓값(max)"으로 합친다.
=> 두 모델 중 하나라도 위험하다고 판단하면 risk_score가 높아지는
   재현율(recall) 우선 전략이다.
   (DLP 특성상 '놓치는 것'이 '오탐'보다 훨씬 치명적이기 때문)

CensorRuntime(censor_runtime_interface.py) 이 이 클래스의
predict_proba() 를 그대로 호출할 수 있도록 sklearn 스타일 인터페이스를 맞춘다.
"""
from typing import List

import numpy as np


class EnsembleCensorModel:
    def __init__(self, ml_model, dl_model, svd, class_names: List[str]):
        self.ml_model = ml_model      # OneVsRestClassifier(SGDClassifier), sparse X 입력
        self.dl_model = dl_model      # MLPClassifier, SVD로 축소된 dense 입력
        self.svd = svd                # TruncatedSVD, dl_model 입력 전처리용
        self.class_names = list(class_names)

    def _ml_proba(self, X_sparse) -> np.ndarray:
        """OneVsRestClassifier -> (N, n_classes) 확률 행렬"""
        proba = self.ml_model.predict_proba(X_sparse)
        return np.asarray(proba)

    def _dl_proba(self, X_sparse) -> np.ndarray:
        """SVD 축소 후 MLPClassifier -> (N, n_classes) 확률 행렬
        MLPClassifier는 원본 라벨(0~6, 단일 라벨)로 학습되었으므로
        predict_proba의 컬럼 순서를 self.class_names 순서로 재정렬한다.
        """
        X_red = self.svd.transform(X_sparse)
        proba = self.dl_model.predict_proba(X_red)
        # self.dl_model.classes_ 는 정수 라벨(0~6)이며 class_names 인덱스와 1:1 대응됨
        ordered = np.zeros((X_red.shape[0], len(self.class_names)), dtype=float)
        for col_idx, raw_label in enumerate(self.dl_model.classes_):
            ordered[:, int(raw_label)] = proba[:, col_idx]
        return ordered

    def predict_proba(self, X_sparse) -> np.ndarray:
        """CensorRuntime.predict_from_features 가 호출하는 표준 인터페이스.
        두 서브모델의 확률 중 클래스별 최댓값을 취한다 (재현율 우선)."""
        p_ml = self._ml_proba(X_sparse)
        p_dl = self._dl_proba(X_sparse)
        return np.maximum(p_ml, p_dl)

    def predict(self, X_sparse):
        proba = self.predict_proba(X_sparse)
        idx = proba.argmax(axis=1)
        return np.array([self.class_names[i] for i in idx])
