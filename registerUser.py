import cv2
import os
import init


def registUser():
    try:
        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        user_id = input("\n 유저 ID를 입력해주세요. ==> ")

        if(os.path.exists(F"dataset/{user_id}")):
            print("이미 등록된 ID입니다. 다시 입력해주세요.")
            user_id = input('\n 유저 ID를 입력해주세요. ==>  ')

        init.createFolder(F"dataset/{user_id}")

        print("\n 얼굴 사진을 저장중입니다... 카메라를 응시해 주세요.")

        count = 0
        save_img_num = 30  # 저장할 사진 갯수

        while(True):
            ret, img = cam.read()
            img = cv2.flip(img, -1)  # 라즈베리파이아니면 제거 (영상 뒤집는것임)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.3,  # 이미지 스케일
                minNeighbors=10  # face 후보들의 개수
            )
            cv2.imshow("IMG", img)

            for x, y, w, h in faces:
                cv2.rectangle(
                    img, (x, y), (x+w, y+h), (0, 255, 0), 2
                )
                count += 1
                print(F"사진 저장 중.. {count}/{save_img_num}")
                cv2.imwrite(F"dataset/{user_id}/" +
                            str(user_id)+F"_{count}"+".jpg", gray[y:y+h, x:x+w])

            key = cv2.waitKey(50)
            if key == ord('q'):
                break
            if(count >= save_img_num):
                break
        cam.release()
    except Exception as e:
        print(e)

    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    try:
        registUser()
    except Exception as e:
        print(e)
