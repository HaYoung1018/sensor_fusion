import pandas as pd

# CSV 파일 경로
file_path = 'mectraj1126.csv'  # 사용자의 CSV 파일 경로

# CSV 파일 읽기
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"CSV 파일 읽기 실패: {e}")
    exit()

# DetectionTime을 datetime으로 변환
def parse_detection_time(time_str):
    try:
        return pd.to_datetime(time_str, format='%Y%m%d%H%M%S.%f')
    except ValueError:
        return pd.to_datetime(time_str, format='%Y%m%d%H%M%S')

df['Timestamp'] = df['DetectionTime'].apply(parse_detection_time)

# SensorID와 TrackingID를 기준으로 그룹화
grouped = df.groupby(['SensorID', 'trackingID'])

# 각 그룹의 시작 시간, 종료 시간, 지속 시간 계산
results = []
for (sensor_id, tracking_id), group in grouped:
    start_time = group['Timestamp'].min()
    end_time = group['Timestamp'].max()
    duration = (end_time - start_time).total_seconds()  # 지속 시간(초 단위)
    results.append({
        'SensorID': sensor_id,
        'TrackingID': tracking_id,
        'Start Time': start_time,
        'End Time': end_time,
        'Duration (seconds)': duration
    })

# 결과를 DataFrame으로 변환
result_df = pd.DataFrame(results)

# 결과를 CSV 파일로 저장
output_file = 'times.csv'
result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"결과가 '{output_file}' 파일로 저장되었습니다.")

# 결과를 터미널에 출력
print(result_df)
