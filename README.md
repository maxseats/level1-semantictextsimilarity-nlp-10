# **📝 Project 소개**

| 항목 | 내용 |
| --- | --- |
| 프로젝트 주제 | 두 문장이 의미적으로 얼마나 유사한지를 수치화하는 자연어처리 태스크 |
| 프로젝트 구현 내용 | • STS 프로젝트를 통해 기계가 자연어를 어떻게 인식하는지 알아볼 수 있다.<br>• 프로젝트에서 임베딩과 토큰화, pre-trained 모델을 활용하면서 모델의 자연어 처리 성능을 비교해볼 수 있다. |
| 진행 기간 | 2023년 12월 13일 ~ 2023년 12월 21일 |

### ⚙️ 개발 환경 및 협업 환경
> Notion  | 모델 실험 기록, 정보 공유 <br>
Slack     | 데이터 및 소통 창구 <br>
Zoom   | 온라인 회의 장소 <br>
Github  | 최종 코드 병합 <br>
Wandb | 하이퍼 파라미터 파인 튜닝
> 

# **👨‍👩‍👧‍👦 Team & Members** 소개

### 💁🏻‍♂️ Members

| **이재형** | **김민석** | **최새연** | **오태연** | **이상수** | **최예진** |
| --- | --- | --- | --- | --- | --- |
| <img src='https://avatars.githubusercontent.com/u/71856506?v=4' height=100 width=100></img> | <img src='https://avatars.githubusercontent.com/u/63552400?v=4' height=100 width=100></img> | <img src='https://avatars.githubusercontent.com/u/71118045?v=4' height=100 width=100></img> | <img src='https://avatars.githubusercontent.com/u/122765534?v=4' height=100 width=100></img> | <img src='https://github.com/boostcampaitech6/level2-klue-nlp-04/assets/118837517/344540c3-a093-4cb8-a694-61164a7380f8' height=100 width=100></img> | <img src='https://avatars.githubusercontent.com/u/69586041?v=4' height=100 width=100></img> |
|  <a href="https://github.com/jaealways" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a> | <a href="https://github.com/maxseats" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a> |  <a href="https://github.com/new-open" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a> | <a href="https://github.com/ohbigkite" target="https://github.com/ohbigkite"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a> | <a href="https://github.com/SangSusu-git" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a> | <a href="https://github.com/yeh-jeans" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a> |


### 👸🏻 Members' Role

> 전반적인 프로젝트 과정을 모두 경험할 수 있도록 분업을 하여 협업을 진행했으며, 초기 개발 환경 구축 단계부터 Github을 연동하여 세부 task 마다 issue와 branch를 생성하여 각자의 task를 수행했다. 이를 통해 코드 버전 관리와 코드 공유가 원활하게 이루어질 수 있었다.
> 

| 이름 | 역할 |
| --- | --- |
| **이재형** | 데이터셋 리서치, Back-translation, EDA 데이터 증강, Loss Function 리서치 및 구현 |
| **김민석** | 문장 도치 증강 및 데이터 불균형 해소 시도, 앙상블 모델 실험, WandB 및 Github 환경 설정 및 연동 코드 추가 |
| **최새연** | 모델 리서치, K-Fold Ensemble / Prediction-Label / Early Stopping / Data Imbalance Sampler 설계 및 실험 |
| **오태연** | EDA 및 데이터 시각화, 모델 및 성능 개선 방법 리서치, 데이터 전처리(오타교정기, 중복문자 제거), 모델 토크나이저 분석, 학습 결과 분석 |
| **이상수** | 데이터 전처리, 앙상블(K-Fold, Soft Voting) 설계 및 실험, difference(predict - label) 도출 및 분석 |
| **최예진** | 데이터 전처리, 데이터 증강, 앙상블 설계 및 실험, 모델 토크나이저 분석, 모델 리서치 |

# 💾 데이터 소개

### **데이터셋 설명**
![image](https://github.com/user-attachments/assets/f761675e-d397-405f-9434-6948acf2f253)

- 기본제공 데이터셋 `train.csv` ,`dev.csv` , `test.csv`
- 국민청원 게시판 제목 데이터, 네이버 영화 감성 분석 코퍼스, 업스테이지(Upstage) 슬랙 데이터
- 총 3가지 출처의 문장 데이터로, 총합 10974 행, train/dev/test 85/5/10 의 비율로 나누어짐.
train ,dev.csv 를 이용하여 모델 학습을 진행함
- 이후 대회 진행중 적용한 전처리 및 증강에 따라 학습을 진행하는 데이터를 변경하여 사용함.

### 데이터셋 통계

- `train.csv` : 총 32470개
- `test_data.csv` : 총 7765개 (정답 라벨 blind = 100으로 임의 표현)

## 💡 Methods

| 분류 | 내용 |
| --- | --- |
| **모델** | • 실험한 모델 : 최종적으로 `klue/roberta-large` 사용<br>`klue/roberta-large`, `klue/roberta-base`,  `klue/roberta-small`, `monologg/koelectra-base-v3-discriminator`, `snunlp/KR-ELECTRA-discriminator`, `beomi/kcbert`, `xlm/roberta-large`, `kykim/bert-kor-base`, `kykim/electra-kor-base` <br>• LSTM layer 추가 : Classification 단계에서 LSTM layer를 추가해줌으로써 일부 토큰의 결과 벡터만을 사용하던 기존 구조 개선, 문장 전체 벡터를 활용할 수 있는 LSTM layer를 추가 |
| **데이터 전처리** | • (Typed) Entity Marker : entity의 위치 정보를 marker로 제공하고 entity의 유형을 제공해서 학습 성능 향상을 시도<br>• 데이터 Query 추가하기 : BERT의 QA Task 학습 방식을 적용하고자 함- sentence 앞 부분에 질문 형태의 쿼리 추가 (예시 : [SUB]와 [OBJ]의 관계는 무엇인가? [SEP] [sentence] [SEP])<br>• Source 스페셜 토큰 추가 : 소스별 타겟값의 분포가 다른 것을 확인, 쿼리문 앞에 3가지 소스 스페셜 토큰을 추가해줌 - [W_PED],[W_TR], [POL]<br>• 한자 제거 : 토큰 결과의 UNK 최소화를 위함. 가장 많이 UNK로 토큰화되었던 한자어 제거 |
| **데이터 증강 및 조정** | • Label Reverse 증강 : 서로 상충되는 의미의 라벨과, subject와 object를 바꿔도 괜찮은 라벨의 경우 subject와 object를 반대로 swap하여 데이터 증강, 10939개의 데이터 증가<br>• Back-Translation 증강 : GoogleTrans 라이브러리를 활용해 문장을 영어로 번역한 후, 이를 다시 한국어로 번역하여 데이터 증강<br>• MLM 증강 :  BERT 기반 모델들의 MLM 학습 방식에서 착안, [MASK] 부분이 기존 문장과 다른 새로운 token으로 패러프레이징 될 것임을 가정하고,증강에 활용 |
| **아키텍쳐 보완** | 1. 과적합 방지<br>• Early Stopping : patience 조정<br>• Hyperparameter Tuning : epoch, learning_rate, batch_size, load_best_model 등<br>2. 불포 불균형 해결<br>• binning 모델링<br>• 특정 라벨 증강 시도 및 no_relation 라벨 undersamping<br>• source별 불균형 해소 시도<br>• Loss Function 변경 (Focal Loss) |
| **검증 전략** | • 9:1, 8:2, 95:5 비율과 random, stratify의 방식으로 valid set 생성해서 평가 <br>• 최종적으로 리더보드에 제출하여 모델 성능 검증<br>• Valid set에 대한 predict 값과 정답값을 비교하는 difference.csv 파일 및 히트맵을 생성하여 정성평가 |
| **앙상블 방법** | • 데이터 전처리와 모델링 기법, 증강 데이터 적용 후 학습한 모델 중 가장 성능이 좋은 모델 10개를 선정하여 soft voting 앙상블을 진행<br>• 성능이 좋은 모델들 중 최대한 다양한 b종류의 모델과 여러 데이터셋이 섞이도록 Soft Voting, Weighted Voting 진행<br>• 성능 개선 : micro f1 75.1084(단일모델 최고) →76.4576 (앙상블) |
