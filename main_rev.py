import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
import random
import threading
import time
from queue import Queue
import matplotlib.pyplot as plt
import csv

#중복데이터 퓨전 --> 선으로 연결
# 가상의 센서 객체 클래스 정의 (카메라 및 라이다)
class SensorObject:
    def __init__(self, position, velocity, direction, acceleration):
        self.position = position  # 객체의 위치 (2D 좌표)
        self.velocity = velocity  # 객체의 속도
        self.direction = direction  # 객체의 진행 방향
        self.acceleration = acceleration  # 객체의 가속도

    # 센서 데이터에 잡음을 추가하는 함수
    def add_noise(self):
        noise_position = self.position + np.random.normal(0, 0.001, 2)  # 위치에 랜덤 잡음 추가
        noise_velocity = self.velocity + random.uniform(-0.005, 0.005)  # 속도에 랜덤 잡음 추가
        return SensorObject(noise_position, noise_velocity, self.direction, self.acceleration)


# 확장 칼만 필터 클래스 정의
class ExtendedKalmanFilter:
    def __init__(self, dt):
        self.ekf = KalmanFilter(dim_x=4, dim_z=2)  # 상태 벡터 크기 4, 측정 벡터 크기 2로 설정
        # 상태 전이 행렬 정의 (dt는 시간 간격)
        self.ekf.F = np.array([[1, 0, dt, 0],
                               [0, 1, 0, dt],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
        # 측정 행렬 정의
        self.ekf.H = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]])
        # 측정 잡음 공분산 행렬 정의 (측정 신뢰도에 기반하여 설정)
        self.ekf.R = np.array([[0.02, 0],
                               [0, 0.02]])
        # 초기 상태 공분산 행렬 설정 (초기 상태의 불확실성 반영)
        self.ekf.P = np.eye(4) * 0.5
        # 과정 잡음 공분산 행렬 설정 (잡음 크기 조정)
        self.ekf.Q = np.eye(4) * 0.001
        self.dt = dt

    # 예측 및 업데이트 수행
    def predict_and_update(self, sensor_object):
        z = np.array([sensor_object.position[0], sensor_object.position[1]])  # 측정 벡터 생성
        self.ekf.predict()  # 예측 단계 수행
        self.ekf.update(z)  # 업데이트 단계 수행
        # 필터링된 상태를 반환
        return np.hstack((self.ekf.x[:2].flatten(), [sensor_object.velocity], [sensor_object.direction],
                          [sensor_object.acceleration]))


# 무향 칼만 필터를 사용한 데이터 융합 함수
def fusion_process(fusion_data_list):
    # 상태 전이 함수 정의
    def fx(x, dt):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        return np.dot(F, x)

    # 측정 함수 정의
    def hx(x):
        return x[:2]

    # 무향 칼만 필터 설정
    points = MerweScaledSigmaPoints(4, alpha=0.1, beta=2., kappa=1.)
    ukf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=0.05, fx=fx, hx=hx, points=points)
    ukf.R = np.array([[0.02, 0], [0, 0.02]])  # 측정 잡음 공분산 행렬 설정
    ukf.Q = np.eye(4) * 0.003  # 과정 잡음 공분산 행렬 설정
    ukf.P = np.eye(4) * 5  # 초기 상태 공분산 행렬 설정
    # 융합 데이터 순차 처리
    for data in fusion_data_list:
        z = np.array([data[0], data[1]])  # 측정 벡터 생성
        ukf.update(z)  # 측정 업데이트 수행
        ukf.predict()  # 예측 수행
    # 평균을 내서 융합 결과 반환
    return np.hstack((
        ukf.x[:2].flatten(),
        np.average([d[2] for d in fusion_data_list], weights=[1 / (d[2] + 1e-6) for d in fusion_data_list]),
        np.average([d[3] for d in fusion_data_list], weights=[1 / (d[3] + 1e-6) for d in fusion_data_list]),
        np.average([d[4] for d in fusion_data_list], weights=[1 / (d[4] + 1e-6) for d in fusion_data_list])
    ))


# 가상의 궤적 데이터 생성 함수
def generate_trajectory_data(num_points):
    trajectory_data = []
    position = np.array([0.0, 0.0])  # 초기 위치
    velocity = 2.0  # 초기 속도
    direction = 0.0  # 초기 방향
    acceleration = 0.1  # 초기 가속도
    offset = np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])  # 위치 오프셋
    # 주어진 점 개수만큼 데이터 생성
    for i in range(num_points):
        dt = random.uniform(0.03, 0.06)  # 시간 간격 랜덤 설정
        # 곡선 형태의 궤적 생성
        position = position + np.array(
            [velocity * dt * (1 + 0.1 * np.sin(i * 0.03)), velocity * dt * (1 + 0.1 * np.cos(i * 0.03))])
        position = position + np.random.normal(0, 0.001, 2)  # 위치에 잡음 추가
        noisy_position = position + offset  # 일정한 위치 오프셋 적용
        trajectory_data.append((SensorObject(noisy_position, velocity, direction, acceleration), dt))
    return trajectory_data


# 카메라 및 라이다 궤적 데이터 생성
num_points = 1000
cam_trajectory = generate_trajectory_data(num_points)  # 카메라 데이터 생성
lidar_trajectory = generate_trajectory_data(num_points)  # 라이다 데이터 생성
# 큐를 사용하여 센서 데이터를 저장
cam_queue = Queue()
lidar_queue = Queue()
# EKF 필터 초기화
ekf_cam = ExtendedKalmanFilter(0.05)
ekf_lidar = ExtendedKalmanFilter(0.05)
# 센서 데이터 출력을 위한 리스트 초기화
cam_positions = []
lidar_positions = []
# 쓰레드를 멈추기 위한 플래그
stop_threads = threading.Event()


# 센서 데이터를 필터링하는 쓰레드 함수
def sensor_thread(sensor_data, queue, ekf_filter, positions_list):
    last_time = time.time()  # 마지막 실행 시간 기록
    while not stop_threads.is_set():  # 쓰레드가 중지되지 않는 한 계속 실행
        current_time = time.time()
        if current_time - last_time >= 0.05:  # 0.05초마다 실행
            if len(sensor_data) == 0:  # 데이터가 더 이상 없으면 쓰레드 중지
                stop_threads.set()
                break
            obj, dt = sensor_data.pop(0)  # 센서 데이터에서 하나 가져오기
            try:
                pred = ekf_filter.predict_and_update(obj)  # 필터링 수행
                queue.put((pred, current_time))  # 큐에 예측 결과와 타임스탬프 저장
                positions_list.append(pred)  # 결과를 리스트에 추가
            except Exception as e:
                print(f"Error in sensor_thread: {e}")
            last_time = current_time  # 마지막 실행 시간 업데이트
        time.sleep(0.01)  # CPU 사용량 줄이기 위해 대기


# 융합 데이터를 처리하는 쓰레드 함수
fused_positions = []


# 융합된 궤적을 부드럽게 처리하는 함수
def final_filter(fused_positions):
    filtered_positions = []
    for i in range(3, len(fused_positions)):
        # 가중 평균을 사용하여 부드러운 궤적 생성
        filtered_position = (
                    0.4 * fused_positions[i] + 0.3 * fused_positions[i - 1] + 0.2 * fused_positions[i - 2] + 0.1 *
                    fused_positions[i - 3])
        filtered_positions.append(filtered_position)
    return np.array(filtered_positions)


# 융합 쓰레드 함수
def fusion_thread():
    fusion_interval = 0.05  # 융합 주기
    fusion_data_list = []  # 융합 데이터를 저장할 리스트
    last_fusion_time = time.time()
    while not stop_threads.is_set():
        current_time = time.time()
        # 카메라 데이터 처리
        while not cam_queue.empty():
            data, timestamp = cam_queue.get()
            if current_time - timestamp <= 0.05:  # 유효한 데이터인지 확인
                fusion_data_list.append(data)
        # 라이다 데이터 처리
        while not lidar_queue.empty():
            data, timestamp = lidar_queue.get()
            if current_time - timestamp <= 0.05:  # 유효한 데이터인지 확인
                fusion_data_list.append(data)
        # 융합 주기마다 융합 수행
        if current_time - last_fusion_time >= fusion_interval:
            if fusion_data_list:
                try:
                    fused_position = fusion_process(fusion_data_list)  # 융합 수행
                    fused_positions.append(fused_position)  # 결과 추가
                except Exception as e:
                    print(f"Error in fusion_thread: {e}")
                fusion_data_list = []  # 융합 리스트 초기화
            last_fusion_time = current_time
        time.sleep(0.01)  # CPU 사용량 줄이기 위해 대기


# 센서 데이터 쓰레드 시작
cam_thread = threading.Thread(target=sensor_thread, args=(cam_trajectory, cam_queue, ekf_cam, cam_positions))
lidar_thread = threading.Thread(target=sensor_thread, args=(lidar_trajectory, lidar_queue, ekf_lidar, lidar_positions))
# 융합 쓰레드 시작
fusion_thread = threading.Thread(target=fusion_thread)
# 쓰레드 실행
cam_thread.start()
lidar_thread.start()
fusion_thread.start()
# 쓰레드가 완료될 때까지 대기
cam_thread.join()
lidar_thread.join()
fusion_thread.join()
# 부드러운 궤적 생성
smoothed_positions = final_filter(fused_positions)

# 결과 시각화
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 타임스탬프 생성
timestamps_cam = np.linspace(0, len(cam_positions) * 0.05, len(cam_positions))
timestamps_lidar = np.linspace(0, len(lidar_positions) * 0.05, len(lidar_positions))
timestamps_smoothed = np.linspace(0, len(smoothed_positions) * 0.05, len(smoothed_positions))

# 카메라 데이터 시각화 (3차원 그래프)
ax.scatter([pos[0] for pos in cam_positions], [pos[1] for pos in cam_positions], timestamps_cam, label='Camera', alpha=0.5, marker='o', color='blue')

# 라이다 데이터 시각화 (3차원 그래프)
ax.scatter([pos[0] for pos in lidar_positions], [pos[1] for pos in lidar_positions], timestamps_lidar, label='Lidar', alpha=0.5, marker='o', color='green')

# 부드러운 융합 결과 시각화 (3차원 그래프)
ax.scatter([pos[0] for pos in smoothed_positions], [pos[1] for pos in smoothed_positions], timestamps_smoothed, label='Smoothed', alpha=0.7, marker='o', color='red')

# 축 설정
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Time')
ax.legend()
ax.set_title('Sensor Fusion Trajectories')
plt.show()



# 결과를 CSV 파일로 저장
with open('sensor_fusion_results.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Camera_X', 'Camera_Y', 'Lidar_X', 'Lidar_Y', 'Fused_X', 'Fused_Y'])
    for cam, lidar, fused in zip(cam_positions, lidar_positions, smoothed_positions):
        csv_writer.writerow([cam[0], cam[1], lidar[0], lidar[1], fused[0], fused[1]])
