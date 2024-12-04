import pandas as pd

# CSV 파일 경로
times_file = 'times.csv'  # 타임 데이터 파일
ground_truth_file = 'rtktraj1126.csv'  # 그라운드 트루 데이터 파일

# CSV 파일 읽기
try:
    times_df = pd.read_csv(times_file)
    ground_truth_df = pd.read_csv(ground_truth_file)
except Exception as e:
    print(f"CSV 파일 읽기 실패: {e}")
    exit()

# Ground Truth 데이터에서 Timestamp 생성
ground_truth_df['Timestamp'] = pd.to_datetime(
    ground_truth_df['ymd'] + ' ' + ground_truth_df['hms']
)

# Ground Truth의 최소 및 최대 시간 계산
ground_truth_min = ground_truth_df['Timestamp'].min()
ground_truth_max = ground_truth_df['Timestamp'].max()

# Start Time과 End Time을 datetime으로 변환
times_df['Start Time'] = pd.to_datetime(times_df['Start Time'])
times_df['End Time'] = pd.to_datetime(times_df['End Time'])

# 결과 저장용 리스트
results = []

# 각 SensorID와 TrackingID에 대해 Ground Truth 범위와 비교
for _, row in times_df.iterrows():
    sensor_id = row['SensorID']
    tracking_id = row['TrackingID']
    start_time = row['Start Time']
    end_time = row['End Time']

    # Start Time과 End Time이 Ground Truth 범위 안에 있는지 확인
    is_start_within = ground_truth_min <= start_time <= ground_truth_max
    is_end_within = ground_truth_min <= end_time <= ground_truth_max

    # 결과 저장
    results.append({
        'SensorID': sensor_id,
        'TrackingID': tracking_id,
        'Start Time': start_time,
        'End Time': end_time,
        'Start Within Range': is_start_within,
        'End Within Range': is_end_within,
        'Both Within Range': is_start_within and is_end_within  # 둘 다 포함되는지 여부
    })

# 결과를 DataFrame으로 변환
result_df = pd.DataFrame(results)

# 결과를 CSV 파일로 저장
output_file = 'asdfghjk.csv'
result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"결과가 '{output_file}' 파일로 저장되었습니다.")

# 결과를 터미널에 출력
print(result_df)
