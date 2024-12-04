import pandas as pd
import folium

# CSV 파일 경로
csv_file_path = 'EKF_filtered.csv'

# 필터링할 SensorID
selected_sensor_ids = ["102030203.0", "102040204.0"]

# 데이터 읽기
try:
    data = pd.read_csv(csv_file_path, dtype={'SensorID': str})
except Exception as e:
    print(f"CSV 파일 읽기 실패: {e}")
    exit()

# SensorID 필터링
filtered_data = data[data['SensorID'].isin(selected_sensor_ids)]

if filtered_data.empty:
    print("선택된 SensorID에 해당하는 데이터가 없습니다.")
    exit()

# TrackingID별로 그룹화
grouped_by_tracking = filtered_data.groupby('trackingID')

# FeatureGroup 생성
layers = []
for tracking_id, tracking_group in grouped_by_tracking:
    feature_group = folium.FeatureGroup(name=f"TrackingID: {tracking_id}")

    for sensor_id, sensor_group in tracking_group.groupby('SensorID'):
        for _, row in sensor_group.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=1,
                color='blue' if sensor_id == selected_sensor_ids[0] else 'red',
                popup=(
                    f"SensorID: {row['SensorID']}<br>"
                    f"TrackingID: {row['trackingID']}<br>"
                )
            ).add_to(feature_group)

    layers.append(feature_group)

# 30개씩 묶어서 지도 생성 및 저장
chunk_size = 20
for i in range(0, len(layers), chunk_size):
    m = folium.Map(location=[filtered_data['Latitude'].mean(), filtered_data['Longitude'].mean()], zoom_start=19)

    for layer in layers[i:i + chunk_size]:
        layer.add_to(m)

    folium.LayerControl().add_to(m)
    map_file = f'map_filtered{i // chunk_size + 1}.html'
    m.save(map_file)
    print(f"지도 파일 '{map_file}'이(가) 저장되었습니다.")
