import pandas as pd
import os

label_csv = "/home/sliuau/video_classification/OTR/otr_video_label/final_label_HD.csv"
split_root = "/home/sliuau/video_classification/OTR/secondFeature_ready/csv_files"

label_df = pd.read_csv(label_csv)

split_dict = {}
for split in os.listdir(split_root):
    for fname in os.listdir(split_root + "/" + split):
        split_dict[fname[:fname.find(".")]] = split

split_list = []
for idx, d in label_df.iterrows():
    if d['ClipID'] not in split_dict:
        split_dict[d['ClipID']] = "Miss"
    split_list.append(split_dict[d['ClipID']])

label_df['Split'] = split_list

label_df.to_csv("label_split.csv")





