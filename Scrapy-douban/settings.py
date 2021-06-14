# Scrapy settings for douban project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'douban'

SPIDER_MODULES = ['douban.spiders']
NEWSPIDER_MODULE = 'douban.spiders'


# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'douban (+http://www.yourdomain.com)'
# 修改编码为utf-8
FEED_EXPORT_ENCODING = 'utf-8'
# Obey robots.txt rules
ROBOTSTXT_OBEY = False




#多线程
# Configure maximum concurrent requests performed by Scrapy (default: 16)
#1 # Downloader最大并发请求下载数量
# CONCURRENT_REQUESTS = 100

# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
DOWNLOAD_DELAY = 1
# The download delay setting will honor only one of:
# 1
# #每个目标域名最大的并发请求数量
# CONCURRENT_REQUESTS_PER_DOMAIN = 100
# #	每个目标IP最大的并发请求数量
# CONCURRENT_REQUESTS_PER_IP = 30

# Disable cookies (enabled by default)
COOKIES_ENABLED = False  #禁用cookie

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
DEFAULT_REQUEST_HEADERS = {
  'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
  'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36',
'cookie':'douban-fav-remind=1; ll="118281"; bid=4cmHmb3QsEk; __yadk_uid=DPqCHmUjUscvG4JdfSp5hr0GWLrZ3KZk; _vwo_uuid_v2=DB1AEDC796997EF36015B64DF8AE9D655|8afe10bb040089321ba1423c35c0a579; push_noty_num=0; push_doumail_num=0; __gads=ID=b40d6a34317675c2-220911f9d6c4003b:T=1605866432:RT=1605866432:R:S=ALNI_MZKBDRhflAlFT-nNwOeTsRqQCcPgA; dbcl2="194149473:jHJ310O9fB0"; __utmz=223695111.1608430821.12.4.utmcsr=douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/; ck=Pwqt; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1608447782%2C%22https%3A%2F%2Fwww.douban.com%2F%22%5D; _pk_ses.100001.4cf6=*; ap_v=0,6.0; __utma=223695111.163452054.1555836815.1608430821.1608447782.13; __utmb=223695111.0.10.1608447782; __utmc=223695111; ct=y; __utma=30149280.1608240005.1608448008.1608448008.1608448008.1; __utmc=30149280; __utmz=30149280.1608448008.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); __utmt_douban=1; __utmb=30149280.18.10.1608448008; _pk_id.100001.4cf6=ecfbd65a8bde7293.1555836815.13.1608450473.1608431594.'

}

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    'douban.middlewares.DoubanSpiderMiddleware': 543,
#}

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html

DOWNLOADER_MIDDLEWARES = {
   'douban.middlewares.DoubanDownloaderMiddleware': 543,
    # 'douban.middlewares.UserAgentDownloadMiddlerware': 543,
   # 'scrapy.contrib.downloadermiddleware.useragent.UserAgentMiddleware': None,
   #  'scrapy.contrib.downloadermiddleware.httpproxy.HttpProxyMiddleware': None,
    # 'douban.middlewares.ProxyMiddleware': 700,
    # 'douban.middlewares.DoubanSpiderMiddleware': 543
    #数字越小，优先级越高

}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
#}

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
   'douban.pipelines.DoubanPipeline': 300,
}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
#AUTOTHROTTLE_ENABLED = True
# The initial download delay
#AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
#AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = 'httpcache'
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'
