import pandas as pd

file_path = "/content/drive/MyDrive/Colab_Notebooks/NMT/additional_data/data/"

news_data = pd.read_csv(file_path + "news1_en_ko.csv")
talking_data = pd.read_csv(file_path + "talking_en_ko.csv")

news_file_train = news_data.sample(frac=0.9, random_state=2022).sort_index()
news_file_dev = news_data.drop(news_file_train.index)

talking_file_train = talking_data.sample(frac=0.7, random_state=2022).sort_index()
talking_file_dev = talking_data.drop(news_file_train.index)

news_file_train.to_csv("news_train.csv", sep=",", na_rep="NaN", index=False)
news_file_dev.to_csv("news_dev.csv", sep=",", na_rep="NaN", index=False)
talking_file_train.to_csv("talking_train.csv", sep=",", na_rep="NaN", index=False)
talking_file_dev.to_csv("talking_dev.csv", sep=",", na_rep="NaN", index=False)

print("done")
