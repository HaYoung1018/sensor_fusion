import pandas as pd
import folium

# CSV 파일 읽기
csv_file = "rtktraj1126.csv"
data = pd.read_csv(csv_file)

# Latitude와 Longitude 열 확인
latitude_column = " Latitude"
longitude_column = " Longitude"

# 지도 초기화 (첫 좌표로 중심 설정)
map_center = [data[latitude_column].iloc[0], data[longitude_column].iloc[0]]
mymap = folium.Map(location=map_center, zoom_start=18)

# 각 좌표를 초록색 점으로 지도에 추가
for idx, row in data.iterrows():
    folium.CircleMarker(
        location=[row[latitude_column], row[longitude_column]],
        radius=1,  # 점의 크기
        color='green',  # 테두리 색상
        fill=True,
        fill_color='green',  # 내부 색상
        fill_opacity=0.8,
    ).add_to(mymap)

# 지도 저장 또는 표시
mymap.save("groundtruth_map.html")
print("지도 파일 'groundtruth_map.html'로 저장되었습니다. 브라우저에서 확인하세요.")
