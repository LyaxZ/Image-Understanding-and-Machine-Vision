# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR, BRAND_MAP_6
from baidu_crawler_patched import PatchedBaiduCrawler

CAR_MODELS = list(BRAND_MAP_6.keys())
NUM_PER_MODEL = 150
THREADS = 4

def main():
    for model in CAR_MODELS:
        brand = BRAND_MAP_6[model]
        save_dir = os.path.join(RAW_DIR, brand)
        os.makedirs(save_dir, exist_ok=True)
        crawler = PatchedBaiduCrawler(storage={'root_dir': save_dir}, downloader_threads=THREADS)
        crawler.crawl(keyword=f"{model} 外观", max_num=NUM_PER_MODEL, filters={'size': 'medium'})
        print(f"✅ {model} 完成")
    print("全部完成！")

if __name__ == "__main__":
    main()