import pandas as pd

# CSV 파일 읽기
file_path = "mectraj1126.csv"  # 여기에 파일 경로를 입력하세요
data = pd.read_csv(file_path)

# 트래킹 ID별로 그룹화 후 센서 ID의 고유값 개수 확인
grouped_data = data.groupby('trackingID')

# 트래킹 ID별 센서 ID 고유값 개수 계산
result = {}
for tracking_id, group in grouped_data:
    sensor_counts = group['SensorID'].value_counts()
    if len(sensor_counts) > 1:  # 센서 ID가 여러 개인 경우
        result[tracking_id] = sensor_counts.to_dict()

# 결과 출력
if result:
    for tracking_id, sensor_info in result.items():
        print(f"트래킹 ID {tracking_id}:")
        for sensor_id, count in sensor_info.items():
            print(f"  - 센서 ID {sensor_id}: {count}개")
else:
    print("트래킹 ID는 같지만 센서 ID가 다른 데이터가 없습니다.")
