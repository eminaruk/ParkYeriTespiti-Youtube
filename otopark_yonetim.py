import cv2
from ultralytics import solutions

kayit = cv2.VideoCapture("parking1.mp4")

if kayit.isOpened() == False:
    print("Video dosyasi acilamadi")
    exit()

genislik = kayit.get(cv2.CAP_PROP_FRAME_WIDTH)
yukseklik = kayit.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = kayit.get(cv2.CAP_PROP_FPS)

video_yazici = cv2.VideoWriter("parking1_yonetim.avi", cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (int(genislik), int(yukseklik)))

park_yonetimi = solutions.ParkingManagement(model = "yolo11n.pt", json_file = "bounding_boxes.json")

while kayit.isOpened():

    durum, kare = kayit.read()

    if durum:

        cikti = park_yonetimi.process_data(kare)

        video_yazici.write(cikti)
    
    else:
        break
    


kayit.release()
video_yazici.release()
cv2.destroyAllWindows()
