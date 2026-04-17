# -*- coding: utf-8 -*-
import time
import random
from icrawler.builtin import BaiduImageCrawler

class PatchedBaiduCrawler(BaiduImageCrawler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Referer': 'https://image.baidu.com/',
        })
    
    def crawl(self, *args, **kwargs):
        # 每次爬取前随机暂停，防止频率过高
        time.sleep(random.uniform(2, 5))
        super().crawl(*args, **kwargs)