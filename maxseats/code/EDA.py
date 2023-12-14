# 데이터 시각화 및 다양한 실험
import pandas as pd
import matplotlib.pyplot as plt


# CSV 파일 읽기
file_path = '../data/train.csv'  # 파일 경로와 이름을 실제 파일 위치에 맞게 수정해주세요.
df = pd.read_csv(file_path)

# 'source'의 각 항목 개수 확인
source_counts = df['source'].value_counts()

# 결과 출력
print(source_counts)
















