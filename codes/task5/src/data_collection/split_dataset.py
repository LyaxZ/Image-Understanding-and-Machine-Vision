# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AUGMENTED_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV

def main():
    data = []
    classes = sorted([d for d in os.listdir(AUGMENTED_DIR) if os.path.isdir(os.path.join(AUGMENTED_DIR, d))])
    label_map = {c: i for i, c in enumerate(classes)}
    for cls in classes:
        cls_path = os.path.join(AUGMENTED_DIR, cls)
        for f in os.listdir(cls_path):
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                data.append([os.path.join(cls, f), label_map[cls]])
    df = pd.DataFrame(data, columns=["path", "label"])
    train, temp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    val, test = train_test_split(temp, test_size=1/3, stratify=temp["label"], random_state=42)
    train.to_csv(TRAIN_CSV, index=False)
    val.to_csv(VAL_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    print(f"类别数：{len(classes)}，映射：{label_map}")
    print(f"训练：{len(train)}，验证：{len(val)}，测试：{len(test)}")

if __name__ == "__main__":
    main()