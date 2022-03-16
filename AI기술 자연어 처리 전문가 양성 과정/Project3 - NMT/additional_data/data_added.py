import csv
import glob

# 경로
file_path = "/content/drive/MyDrive/Colab_Notebooks/NMT/additional_data/data/"
dev_file_list = glob.glob(file_path + "dev/*")
train_file_list = glob.glob(file_path + "train/*")

with open("dev_added.csv", "w") as dev:
    for i, file in enumerate(dev_file_list):
        if i == 0:
            with open(file, "r") as added:
                while True:
                    line = added.readline()

                    if not line:
                        break

                    dev.write(line)

            file_name = file.split("\\")[-1]
            print(file.split("\\")[-1] + " write complete...")

        else:
            with open(file, "r") as added:
                n = 0
                while True:
                    line = added.readline()

                    if n != 0:
                        dev.write(line)

                    if not line:
                        break
                    n += 1

            file_name = file.split("\\")[-1]
            print(file.split("\\")[-1] + " write complete...")

with open("train_added.csv", "w") as train:
    for i, file in enumerate(train_file_list):
        if i == 0:
            with open(file, "r") as added:
                while True:
                    line = added.readline()

                    if not line:
                        break

                    train.write(line)

            file_name = file.split("\\")[-1]
            print(file.split("\\")[-1] + " write complete...")

        else:
            with open(file, "r") as added:
                n = 0
                while True:
                    line = added.readline()

                    if n != 0:
                        train.write(line)

                    if not line:
                        break
                    n += 1

            file_name = file.split("\\")[-1]
            print(file.split("\\")[-1] + " write complete...")
