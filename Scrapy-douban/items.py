# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class DoubanItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    
    movice = scrapy.Field()  # 电影名
    user_name = scrapy.Field()  # 评论者用户 个人空间网页地址
    user_star = scrapy.Field()  # 评论者用户名
    user_review_time = scrapy.Field()  # 评论时间
    user_likes_number = scrapy.Field()  # 评论被点赞数
    user_review = scrapy.Field()  # 评论者 评语

    user_place = scrapy.Field()  # 评论者用户常居城市
    user_registration_time = scrapy.Field()  # 用户注册时间
    user_review_number = scrapy.Field()  # 用户总共评论数量
    user_movice_watching = scrapy.Field()  # 用户在看电影量
    user_movice_want = scrapy.Field()  # 用户在看电影量
    user_movice_watch = scrapy.Field()  # 用户在看电影量
    user_follow = scrapy.Field()  # 用户被关注量

    # url = scrapy.Field()


