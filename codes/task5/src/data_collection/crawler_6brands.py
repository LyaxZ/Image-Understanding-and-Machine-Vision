# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR, BRAND_MAP_6
from icrawler.builtin import BaiduImageCrawler   # 改为百度

CAR_MODELS = list(BRAND_MAP_6.keys())
NUM_PER_MODEL = 300
THREADS = 8

def main():
    for model in CAR_MODELS:
        brand = BRAND_MAP_6[model]
        save_dir = os.path.join(RAW_DIR, brand)
        os.makedirs(save_dir, exist_ok=True)
        crawler = BaiduImageCrawler(storage={'root_dir': save_dir}, downloader_threads=THREADS)
        # 百度爬虫的 filters 只支持 'size', 'color', 'type' 等，这里只用 size
        crawler.crawl(
            keyword=f"{model} 外观 实拍",
            max_num=NUM_PER_MODEL,
            filters={'size': 'medium'}
        )
        print(f"✅ {model} 完成")
    print("全部完成！")

if __name__ == "__main__":
    main()