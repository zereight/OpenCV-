import os
import cv2
import numpy as np
from PIL import Image


# def getImageInfo(path, faceSamples, ids):

#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
#     face_detector = cv2.CascadeClassifier(
#         cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#     for imagePath in imagePaths:
#         if ".DS_Store" in imagePath:  # MAC 아니라면 지워도됨.
#             continue

#         PIL_img = Image.open(imagePath).convert('L')  # onvert it to grayscale
#         img_numpy = np.array(PIL_img, 'uint8')
#         name = int(os.path.split(imagePath)[-1].split(".")[0].split("_")[0])

#         faces = face_detector.detectMultiScale(img_numpy)

#         for (x, y, w, h) in faces:
#             faceSamples.append(img_numpy[y:y+h, x:x+w])
#             ids.append(name)
#     return faceSamples, ids


def Trainer():

    try:
        names = []
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        imageFolders = [os.path.join("dataset", f)
                        for f in os.listdir("dataset")]
        faceSamples = []
        ids = []
        idCnt = 0
        print("\n 학습 중 입니다...")

        for imageFolder in imageFolders:
            if ".DS_Store" in imageFolder:  # MAC 아니라면 지워도됨.
                continue
            name = os.path.split(imageFolder)[-1]
            # faceSamples, ids = getImageInfo(imageFolder, faceSamples, ids)
            for imagePath in sorted([f for f in os.listdir(imageFolder)]):

                if ".DS_Store" in imagePath:  # MAC 아니라면 지워도됨.
                    continue

                PIL_img = Image.open(os.path.join(imageFolder, imagePath)).convert(
                    'L')  # onvert it to grayscale
                img_numpy = np.array(PIL_img, 'uint8')

                faces = face_detector.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y+h, x:x+w])
                    ids.append(idCnt)
                    names.append(name)
            idCnt += 1  # 폴더넘어가면 사람도 넘어가는 거니까 폴더 바뀔때 +1
        recognizer.train(faceSamples, np.array(ids))

        # Save the model into trainer/trainer.yml
        # recognizer.save() worked on Mac, but not on Pi
        try:
            recognizer.write('train_result/trainer.yml')
        except:
            os.remove('train_result/trainer.yml')
            recognizer.write('train_result/trainer.yml')

        # Print the numer of faces trained and end program
        print(f"\n{len(np.unique(ids))} 개 얼굴 학습 완료 ")
        # print(names)
        return names

    except Exception as e:
        print(e)
        # return Trainer()
