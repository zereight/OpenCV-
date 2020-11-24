import os
import init
import registerUser

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

    registerUser.registUser()
