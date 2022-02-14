import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN

def main():
    num_faces_train = count_num_faces("train")
    num_faces_train.to_csv("data/num_faces_train.csv", index = False, header = True)
    num_faces_test = count_num_faces("test")
    num_faces_test.to_csv("data/num_faces_test.csv", index = False, header = True)

def count_num_faces(name):
    data = pd.read_csv(f"data/{name}.csv")
    mtcnn = MTCNN(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    num_faces = {"LOAN_ID": [], "NUM_FACES": []}
    for index, row in tqdm(data.iterrows(), total = data.shape[0]):
        image_path = Path("data/{}_images/{}.jpg".format(name, row["IMAGE_ID"]))
        num_faces["LOAN_ID"].append(row["LOAN_ID"])
        if image_path.exists():
            image = Image.open(image_path)
            boxes, probs = mtcnn.detect(image)
            if boxes is not None:
                num_faces["NUM_FACES"].append(boxes.shape[0])
            else:
                num_faces["NUM_FACES"].append(0)
        else:
            num_faces["NUM_FACES"].append(0)
    num_faces = pd.DataFrame(num_faces)

    return num_faces

if __name__ == "__main__":
    main()
