# Baseline RL Recommender

Tabular Q-Learning 기반의 간단한 음악 추천 베이스라인 스크립트입니다. 사용자가 직전에 들은 곡 한 곡만 상태(state)로 보고, Top‑K 빈도 상위 곡들 중에서 다음 곡을 추천하도록 학습합니다.

## 파일 구성

* `baseline_rl_training.py`

  * `listening_history.csv`(공백 구분)로부터 사용자별 청취 시퀀스를 생성
  * 청취 빈도 상위 `TOP_K`곡 필터링
  * Tabular Q-Learning으로 에피소드별 학습 수행
  * 학습 보상 로그 및 평가 정확도 출력
  * 인기 곡 샘플 추천 결과 출력
  * 최종 Q-테이블(`q_table.pkl`) 저장

## 요구사항

* Python 3.7 이상
* 설치 패키지:

  ```bash
  pip install pandas numpy
  ```

## 사용 방법

1. 리포지토리 루트에 `listening_history.csv` 파일을 복사합니다.
   포맷: `user song YYYY-MM-DD HH:MM` (공백 구분)
2. 스크립트를 실행합니다:

   ```bash
   python baseline_rl_training.py
   ```
3. 학습이 완료되면 다음이 출력됩니다:

   * 에피소드별 총 보상 및 평균 보상
   * 학습 요약(평균 보상 통계)
   * 최종 추천 정확도
   * 인기 곡에 대한 샘플 다음 곡 추천
4. `q_table.pkl` 파일이 생성되며, 이후 재사용 및 분석에 활용할 수 있습니다.

## 주요 하이퍼파라미터

* `TOP_K` (기본: 500): 학습에 사용할 상위 곡 개수 제한
* `EPISODES` (기본: 10): 전체 사용자 데이터에 대한 학습 반복 횟수
* `ALPHA` (기본: 0.1): 학습률
* `GAMMA` (기본: 0.9): 할인율
* `EPSILON` (기본: 0.1): ε-greedy 탐험율

## Implicit Feedback

이 스크립트는 **암묵적 피드백(Implicit Feedback)** 만을 사용하여 학습합니다:

* **Positive signal**: `listening_history.csv`에 기록된 사용자 재생 이벤트
* **Reward 구조**: 에이전트가 추천한 다음 곡이 실제 재생된 곡과 일치하면 `1.0`, 그렇지 않으면 `0.0`
* **Negative signal**: 직접 비례 샘플링 없이, 추천 시 불일치한 항목은 자연스럽게 negative 예시로 작동

## 출력 파일

* `q_table.pkl`: 학습된 Q-테이블(`float32` 배열)과 아이템-인덱스 매핑 정보(`item2idx`, `idx2item` 딕셔너리)
