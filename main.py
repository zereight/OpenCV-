import os
import init
import registerUser
import trainer
import detection

data = dict()


def sync_dataset():  # 폴더들에 있는 데이터들을 읽어옵니다.
    for idx, p in enumerate(os.listdir("dataset")):
        data[idx] = p


if __name__ == "__main__":
    # 필요한 폴더 없으면 생성
    if(not os.path.exists("dataset")):
        init.createFolder("dataset")
    if(not os.path.exists("train_result")):
        init.createFolder("train_result")

    # 폴더가 있다면 데이터셋 동기화
    sync_dataset()

    while(True):
        try:
            order = input("""
명령할 동작을 입력해주세요.
    1: 유저 등록
    2: 감지
    3: 초기화
    4: 학습
    0: 종료
""")
            if(order == "1"):
                registerUser.registUser()
            elif(order == "2"):
                detection.detecting(trainer.trainer())
            elif(order == "3"):
                init.allClear()
                init.init()
                print("초기화를 완료했습니다.\n")
            elif(order == "4"):
                print(trainer.trainer())
            elif(order == "0"):
                break
            else:
                print("\n잘못입력하셨습니다.")
        except Exception as e:
            print(e)
