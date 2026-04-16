# -*- coding: utf-8 -*-
"""
车型图片爬虫（百度加速版）
- 使用百度图片搜索引擎
- 多线程下载，速度提升
- 支持中文路径，稳定可靠
"""

import os
from icrawler.builtin import BaiduImageCrawler

# 配置
BASE_DIR = r"D:\College\大三下\图像理解与机器视觉\Expe\codes\task5"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

CAR_MODELS = [
    "大众朗逸", "大众速腾", "大众帕萨特",
    "丰田卡罗拉", "丰田凯美瑞",
    "本田思域", "本田雅阁",
    "宝马3系", "宝马5系",
    "奔驰C级", "奔驰E级", "奔驰S级",
    "比亚迪秦", "比亚迪汉"
]

NUM_IMAGES = 300          # 每类图片数量
DOWNLOAD_THREADS = 8      # 下载线程数，建议5~10

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"图片保存路径：{RAW_DIR}\n")

    for model in CAR_MODELS:
        print(f"开始爬取：{model}")
        save_dir = os.path.join(RAW_DIR, model.replace(" ", "_"))
        os.makedirs(save_dir, exist_ok=True)

        # 使用百度爬虫，设置下载线程数
        crawler = BaiduImageCrawler(
            storage={'root_dir': save_dir},
            downloader_threads=DOWNLOAD_THREADS
        )

        keyword = f"{model} 外观 街拍"

        try:
            crawler.crawl(
                keyword=keyword,
                max_num=NUM_IMAGES,
                filters={'size': 'medium'}
            )
            print(f"✅ {model} 完成\n")
        except Exception as e:
            print(f"❌ {model} 失败：{e}\n")

    print("所有车型爬取完毕！")

if __name__ == "__main__":
    main()