import pandas as pd
from geopy.distance import geodesic

# 파일 경로 설정
ekf_file = 'EKF_filtered.csv'
rtk_file = 'rtktraj1126.csv'

# 데이터 읽기
try:
    ekf_data = pd.read_csv(ekf_file)
    rtk_data = pd.read_csv(rtk_file)
except Exception as e:
    print(f"CSV 파일 읽기 실패: {e}")
    exit()

# 데이터 확인
if ekf_data.empty or rtk_data.empty:
    print("EKF 또는 RTK 데이터가 비어 있습니다.")
    exit()

# RTK 데이터에서 ymd와 hms 결합하여 시간 열 생성
rtk_data['Timestamp'] = pd.to_datetime(rtk_data['ymd'] + ' ' + rtk_data['hms'])

# DetectionTime 포맷 처리
def parse_detection_time(time_str):
    try:
        # 소수점 이하 초가 포함된 경우
        return pd.to_datetime(time_str, format='%Y%m%d%H%M%S.%f')
    except ValueError:
        # 소수점 이하 초가 없는 경우
        return pd.to_datetime(time_str, format='%Y%m%d%H%M%S')

# EKF 데이터에서 Timestamp 생성
ekf_data['Timestamp'] = ekf_data['DetectionTime'].apply(parse_detection_time)

# EKF 데이터에서 SensorID와 TrackingID를 기준으로 그룹화
ekf_grouped = ekf_data.groupby(['SensorID', 'trackingID'])

# RTK 데이터는 시간 순서대로 정렬
rtk_data = rtk_data.sort_values(by='Timestamp')

# 결과 저장
errors = []
print("\n=== EKF 추정 값과 RTK Ground Truth 비교 ===")

for (sensor_id, tracking_id), ekf_group in ekf_grouped:
    # EKF 데이터를 시간 순서대로 정렬
    ekf_group = ekf_group.sort_values(by='Timestamp')

    # RTK 데이터와 EKF 데이터 매칭
    matched_distances = []
    for _, ekf_row in ekf_group.iterrows():
        ekf_time = ekf_row['Timestamp']
        ekf_coords = (ekf_row['Latitude'], ekf_row['Longitude'])

        # RTK 데이터에서 가장 가까운 시간의 좌표 찾기
        closest_rtk_row = rtk_data.iloc[(rtk_data['Timestamp'] - ekf_time).abs().argsort()[:1]]
        rtk_coords = (closest_rtk_row[' Latitude'].values[0], closest_rtk_row[' Longitude'].values[0])

        # EKF와 RTK 간 거리 계산
        distance = geodesic(ekf_coords, rtk_coords).meters
        matched_distances.append(distance)

    # 평균 오차 계산
    avg_error = sum(matched_distances) / len(matched_distances)
    errors.append((sensor_id, tracking_id, avg_error))
    print(f"SensorID {sensor_id}, TrackingID {tracking_id}: 평균 오차 = {avg_error:.2f}m")
