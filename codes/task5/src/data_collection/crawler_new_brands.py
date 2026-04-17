# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_NEW_DIR, NEW_BRAND_MODELS
from icrawler.builtin import BingImageCrawler

NUM_PER_MODEL = 200
THREADS = 8

def main():
    for brand, models in NEW_BRAND_MODELS.items():
        brand_dir = os.path.join(RAW_NEW_DIR, brand)
        os.makedirs(brand_dir, exist_ok=True)
        for model in models:
            print(f"爬取 {brand} - {model}")
            crawler = BingImageCrawler(
                storage={'root_dir': brand_dir},
                downloader_threads=THREADS,
                parser_threads=2
            )
            crawler.crawl(
                keyword=f"{model} 外观 实拍",
                max_num=NUM_PER_MODEL,
                filters={'size': 'medium'}
            )
        print(f"✅ {brand} 完成")
    print("所有新品牌完成！")

if __name__ == "__main__":
    main()