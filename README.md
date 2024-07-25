# LLM_PEFT_with_Sentiment_Dataset

### Abstract
감성 대화 말뭉치를 LLM에 학습시키고 학습전과 후의 성능 비교를 통해 특정 도메인의 언어를 학습시켰을때 더 정확도가 높아지는 지를 확인한다.

### Description
- Dataset : [감성 대화 말뭉치](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86)
- LLM : gemma:2b-instruct-4Q_0

# 감정 매칭 프로젝트

이 프로젝트는 gemma:2b-instruct LLM이 감정 코퍼스 데이터셋을 사용한 파라미터 효율적 미세 조정(Parameter Efficient Fine Tuning, PEFT)을 수행하기 전후로 입력 문장의 감정을 얼마나 잘 매칭하는지 테스트하는 것을 목표로 합니다.

## 개요

이 프로젝트의 주요 목표는 다음과 같습니다:
1. 입력 문장의 감정을 매칭하는 초기 gemma:2b-instruct LLM의 성능을 평가합니다.
2. 감정 코퍼스 데이터셋을 사용하여 모델을 파라미터 효율적 미세 조정(PEFT)합니다.
3. 미세 조정된 모델의 성능을 재평가하고 초기 결과와 비교합니다.

## 데이터셋

이 프로젝트에서는 감정 코퍼스 데이터셋을 사용합니다. 이 데이터셋은 다양한 문장과 해당 감정 레이블을 포함하고 있어 학습 및 평가에 강력한 기반을 제공합니다.

## 방법론

1. **초기 평가**: 감정 코퍼스 데이터셋에서 gemma:2b-instruct LLM의 성능을 평가합니다.
2. **파라미터 효율적 미세 조정 (PEFT)**: 감정 코퍼스 데이터셋을 사용하여 gemma:2b-instruct LLM을 미세 조정합니다.
3. **조정 후 평가**: 동일한 데이터셋에서 미세 조정된 모델의 성능을 재평가합니다.
4. **비교 및 분석**: 미세 조정 전후의 결과를 비교하여 PEFT의 효과를 확인합니다.

## 프로젝트 구조

- `data/`: 감정 코퍼스 데이터셋을 포함합니다.
- `scripts/`: 데이터 전처리, 모델 학습 및 평가 스크립트를 포함합니다.
- `results/`: 평가 결과를 저장합니다.
- `README.md`: 이 파일입니다.

## 요구 사항

- Python 3.8 이상
- PyTorch
- Hugging Face Transformers
- 기타 필요한 종속성은 `requirements.txt`에 나와 있습니다.

## 설치

저장소를 클론하고 필요한 종속성을 설치합니다:

```bash
git clone https://github.com/yourusername/emotion-matching-project.git
cd emotion-matching-project
pip install -r requirements.txt
