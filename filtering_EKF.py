import pandas as pd
import numpy as np

# CSV 파일 읽기
input_file = "mectraj1126.csv"
output_file = "EKF_filtered.csv"

# 데이터 로드
data = pd.read_csv(input_file)


# EKF 초기화 함수
def initialize_ekf(state, covariance, process_noise, measurement_noise):
    ekf = {
        "state": state,
        "covariance": covariance,
        "process_noise": process_noise,
        "measurement_noise": measurement_noise,
    }
    return ekf


# EKF 업데이트 함수
def ekf_update(ekf, measurement, dt):
    # 상태 및 공분산 가져오기
    x = ekf["state"]
    P = ekf["covariance"]
    Q = ekf["process_noise"]
    R = ekf["measurement_noise"]

    # 예측 단계
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    # 측정값 업데이트 단계
    H = np.eye(4)
    z = measurement
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_update = x_pred + K @ y
    P_update = (np.eye(4) - K @ H) @ P_pred

    # 결과 갱신
    ekf["state"] = x_update
    ekf["covariance"] = P_update
    return ekf


# EKF 초기 값 설정
process_noise = np.diag([0.1, 0.1, 1, 1])  # 프로세스 노이즈
measurement_noise = np.diag([0.5, 0.5, 0.5, 0.5])  # 측정 노이즈
covariance = np.eye(4) * 1.0  # 초기 공분산
filtered_data = []

# 같은 SensorID, TrackingID로 객체 그룹화
grouped = data.groupby(["SensorID", "trackingID"])

for (sensor_id, tracking_id), group in grouped:
    # EKF 초기 상태 설정 (첫 번째 데이터로 초기화)
    first_row = group.iloc[0]
    state = np.array([first_row["Latitude"], first_row["Longitude"], 0, 0])  # 초기 속도는 0으로 설정
    ekf = initialize_ekf(state, covariance, process_noise, measurement_noise)

    for i in range(len(group)):
        row = group.iloc[i]
        if i == 0:
            dt = 0
        else:
            prev_row = group.iloc[i - 1]

            # DetectionTime 포맷 처리
            try:
                current_time = pd.to_datetime(row["DetectionTime"], format='%Y%m%d%H%M%S.%f')
            except ValueError:
                current_time = pd.to_datetime(row["DetectionTime"], format='%Y%m%d%H%M%S')

            try:
                prev_time = pd.to_datetime(prev_row["DetectionTime"], format='%Y%m%d%H%M%S.%f')
            except ValueError:
                prev_time = pd.to_datetime(prev_row["DetectionTime"], format='%Y%m%d%H%M%S')

            dt = (current_time - prev_time).total_seconds()

        measurement = np.array([row["Latitude"], row["Longitude"], 0, 0])
        ekf = ekf_update(ekf, measurement, dt)

        # EKF 결과 저장
        filtered_row = {
            "DetectionTime": row["DetectionTime"],
            "SensorID": row["SensorID"],
            "trackingID": row["trackingID"],
            "objectType": row["objectType"],
            "Latitude": ekf["state"][0],
            "Longitude": ekf["state"][1],
            "Speed": row["Speed"],
            "Heading": row["Heading"],
        }
        filtered_data.append(filtered_row)

# 필터링된 데이터 저장
filtered_df = pd.DataFrame(filtered_data)
filtered_df.to_csv(output_file, index=False)

print(f"필터링된 데이터가 {output_file}에 저장되었습니다.")
