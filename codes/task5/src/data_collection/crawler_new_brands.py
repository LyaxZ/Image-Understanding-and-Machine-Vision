# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_NEW_DIR, NEW_BRAND_MODELS
from baidu_crawler_patched import PatchedBaiduCrawler   # 改用补丁版

NUM_PER_MODEL = 150   # 适当减少数量，降低被封概率
THREADS = 4           # 降低线程数

def main():
    for brand, models in NEW_BRAND_MODELS.items():
        brand_dir = os.path.join(RAW_NEW_DIR, brand)
        os.makedirs(brand_dir, exist_ok=True)
        for model in models:
            print(f"爬取 {brand} - {model}")
            crawler = PatchedBaiduCrawler(
                storage={'root_dir': brand_dir},
                downloader_threads=THREADS,
                parser_threads=1
            )
            # 关键词稍微简化，有时"实拍"会返回空结果
            keyword = f"{model} 外观"
            crawler.crawl(
                keyword=keyword,
                max_num=NUM_PER_MODEL,
                filters={'size': 'medium'}
            )
            print(f"  完成 {model}")
        print(f"✅ {brand} 全部车型完成")
    print("所有新品牌爬取完毕！")

if __name__ == "__main__":
    main()