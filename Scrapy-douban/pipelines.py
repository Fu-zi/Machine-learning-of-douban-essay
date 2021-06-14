# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter

import pymysql.cursors
class DoubanPipeline:
    # 异步机制将数据写入到mysql数据库中
    # pass

    #创建初始化函数，当通过此类创建对象时首先被调用的方法
    def __init__(self):
            # 连接数据库
        self.connect = pymysql.connect(
            host='localhost',  # 数据库地址
            port=3306,  # 数据库端口
            db='douban',  # 数据库名
            user='root',  # 数据库用户名
            passwd='123456',  # 数据库密码
            charset='utf8',  # 编码方式
            use_unicode=True)

        # 通过cursor执行增删查改
        self.cursor = self.connect.cursor()
        print("连接数据库成功")
    def process_item(self, item, spider):
        # 执行插入数据到数据库操作
        self.cursor.execute("""insert into db(movice, user_name, user_star, user_review_time,
        user_likes_number, user_review, user_place, user_registration_time, user_review_number,
        user_movice_watching, user_movice_want, user_movice_watch, user_follow)
            value (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            # item里面定义的字段和表字段对应, 插入值
            (item['movice'], item['user_name'], item['user_star'], item['user_review_time'],
             item['user_likes_number'], item['user_review'], item['user_place'], item['user_registration_time'],
             item['user_review_number'], item['user_movice_watching'], item['user_movice_want'],
             item['user_movice_watch'], item['user_follow']))


        # 提交sql语句
        self.connect.commit()
        return item  # 实现返回


    def close_spider(self, spider):
        # 关闭游标和连接
        self.cursor.close()
        self.connect.close()