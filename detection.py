import cv2
import os


def detecting(models):
    try:
        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        cam = cv2.Videocamture(0)
        cam.set(cv2.cam_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.cam_PROP_FRAME_HEIGHT, 480)
        while True:
            # 카메라로 부터 사진 한장 읽기
            image, frame = cam.read()

            try:
                min_score = 999  # 가장 낮은 점수로 예측된 사람의 점수
                min_score_name = ""  # 가장 높은 점수로 예측된 사람의 이름

                # 검출된 사진을 흑백으로 변환
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 위에서 학습한 모델로 예측시도
                for key, model in models.items():
                    result = model.predict(face)
                    if min_score > result[1]:
                        min_score = result[1]
                        min_score_name = key

                # min_score 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다.
                if min_score < 500:
                    # ????? 어쨋든 0~100표시하려고 한듯
                    confidence = int(100*(1-(min_score)/300))
                    # 유사도 화면에 표시
                    display_string = str(confidence) + \
                        '% Confidence it is ' + min_score_name
                cv2.putText(image, display_string, (100, 120),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)
                # 75 보다 크면 동일 인물로 간주해 UnLocked!
                if confidence > 75:
                    cv2.putText(image, "Unlocked : " + min_score_name,
                                (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Face Cropper', image)
                else:
                    # 75 이하면 타인.. Locked!!!
                    cv2.putText(image, "Locked", (250, 450),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Face Cropper', image)
            except:
                # 얼굴 검출 안됨
                cv2.putText(image, "Face Not Found", (250, 450),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('Face Cropper', image)
                pass
            key = cv2.waitKey(50)
            if key == ord('q'):
                break
    except Exception as e:
        print(e)
    cam.release()
    cv2.destroyAllWindows()
