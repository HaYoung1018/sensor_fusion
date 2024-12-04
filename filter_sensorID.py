import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
import random
import threading
import time
from queue import Queue
import matplotlib.pyplot as plt
import csv

# 확장 칼만 필터 클래스 정의
class ExtendedKalmanFilter:
    def __init__(self, dt):
        self.ekf = KalmanFilter(dim_x=4, dim_z=2)
        self.ekf.F = np.array([[1, 0, dt, 0],
                               [0, 1, 0, dt],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
        self.ekf.H = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]])
        self.ekf.R = np.array([[0.02, 0],
                               [0, 0.02]])
        self.ekf.P = np.eye(4) * 5
        self.ekf.Q = np.eye(4) * 0.001
        self.dt = dt

    def predict_and_update(self, sensor_object):
        z = np.array([sensor_object.position[0], sensor_object.position[1]])
        self.ekf.predict()
        self.ekf.update(z)
        return np.hstack((self.ekf.x[:2].flatten(), [sensor_object.velocity], [sensor_object.direction],
                          [sensor_object.acceleration]))


# 센서 객체 클래스
class SensorObject:
    def __init__(self, position, velocity, direction, acceleration):
        self.position = position
        self.velocity = velocity
        self.direction = direction
        self.acceleration = acceleration


# CSV 데이터 읽기 및 필터링
def load_and_filter_data(csv_path, selected_sensor_ids):
    try:
        # CSV 파일 읽기
        data = pd.read_csv(csv_path)
        # SensorID 필터링
        filtered_data = data[data['SensorID'].isin(selected_sensor_ids)]
        if filtered_data.empty:
            raise ValueError("필터링된 데이터가 없습니다. SensorID를 확인하세요.")
        print(f"필터링된 데이터 개수: {len(filtered_data)}")
        return filtered_data
    except Exception as e:
        print(f"CSV 데이터 로드 또는 필터링 실패: {e}")
        exit()


# 실제 센서 데이터를 EKF에 입력할 수 있는 객체로 변환
def convert_to_sensor_objects(data):
    sensor_objects = []
    for _, row in data.iterrows():
        position = np.array([row['Latitude'], row['Longitude']])
        velocity = row['Speed']
        direction = row['Heading']
        acceleration = 0.1  # 가속도는 임의의 값으로 설정
        sensor_objects.append(SensorObject(position, velocity, direction, acceleration))
    return sensor_objects


# CSV 파일 경로 및 필터링할 SensorID
csv_path = 'trajectory.csv'
selected_sensor_ids = [102030203, 102040204]  # int형 SensorID

# CSV 데이터 읽기 및 필터링
filtered_data = load_and_filter_data(csv_path, selected_sensor_ids)

# 필터링된 데이터를 센서 객체로 변환
sensor_objects = convert_to_sensor_objects(filtered_data)

# EKF 초기화
ekf = ExtendedKalmanFilter(dt=0.05)

# 필터링된 센서 데이터를 EKF로 처리
filtered_positions = []
for obj in sensor_objects:
    pred = ekf.predict_and_update(obj)
    filtered_positions.append(pred)

# 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(filtered_data['Latitude'], filtered_data['Longitude'], label='Original Data', alpha=0.5)
plt.scatter([pos[0] for pos in filtered_positions], [pos[1] for pos in filtered_positions], label='Filtered Data', alpha=0.5)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend()
plt.title('Sensor Data Filtering with EKF')
plt.show()

# 결과를 CSV 파일로 저장
output_path = 'filtered_sensor_data.csv'
with open(output_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Latitude', 'Longitude', 'Velocity', 'Direction', 'Acceleration'])
    for pos in filtered_positions:
        csv_writer.writerow(pos)
print(f"필터링된 데이터가 '{output_path}'에 저장되었습니다.")
