import cv2
import os
import requests
import json
from collections import OrderedDict
from datetime import datetime, timedelta
import timeit
import gc

file_data = OrderedDict()
file_data['time'] = "2020-11-28-11-16"
file_data['temperature'] = [1,18]
file_data['humidity'] = [1, 34]
file_data['gas'] = [1, 20]

headers = {'content-type': 'application/json'}

def face_detecting(img, size=0.5):
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 10)
    if faces is ():
        return img, []
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


def detecting(models):
    try:

        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:

            ret, frame = cam.read()

            image, face = face_detecting(frame)

            try:
                min_score = 999  # 가장 낮은 점수로 예측된 사람의 점수
                min_score_name = ""  # 가장 높은 점수로 예측된 사람의 이름

                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                for key, model in models.items():
                    result = model.predict(face)
                    if min_score > result[1]:
                        min_score = result[1]
                        min_score_name = key

                # min_score 신뢰도, 0에 가까우면 완벽
                if min_score < 500:

                    confidence = int(100*(1-(min_score)/300))
                    # 유사도 화면에 표시
                    display_string = F"{min_score_name} {str(confidence)}%"

                # 가장 높은 유사도의 인물 텍스트로 표시
                cv2.putText(image, display_string, (100, 120),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)

                # 87% 이상이면 감지성공(테스트 결과 87에서 잘걸러내는듯 ㅎ)
                if confidence >= 80:
                    cv2.putText(image, F"{min_score_name} is detected!",
                                (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imwrite(F"{datetime.now()}findFace.jpg", face)
                    files = open(F'{datetime.now()}findFace.jpg', 'rb')
                    upload = {
                        'file': face
                    }
                    data = {
                        'user_id': min_score_name,
                        'datatime': datatime.now()
                    }

                    res = requests.post(
                        'http://192.168.0.196:10023/detectPerson', files=upload, data=json.dumps(data),headers=headers)

                else:  # 87% 이하 감지일때는 아직 잠금해제 안함
                    cv2.putText(image, "Unknown", (250, 450),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    cv2.imwrite(F"{datetime.now()}uknown.jpg", face)
                    files = open(F'{datetime.now()}uknown.jpg', 'rb')
                    upload = {
                        'file': files,
                        'datatime': datatime.now()
                    }
                    data = {'user_id': 'unknown'}

                    res = requests.post(
                        'http://192.168.0.196:10023/detectPerson', files=upload, data=json.dumps(data),headers=headers)

                # res = requests.post(
                #     'http://localhost:10023/file', files=upload)
                cv2.imshow('img', image)

            except:
                # 얼굴 검출 안됨
                cv2.putText(image, "Face detecting...", (250, 450),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

                cv2.imshow('img', image)
            # cv2.imshow('img', image)
            key = cv2.waitKey(50)
            if key == ord('q'):
                break
        cam.release()
    except Exception as e:
        print(e)
        # pass
    cv2.destroyAllWindows()
    cv2.waitKey(1)
