import scrapy
import json
from ..items import DoubanItem
import re


class DbSpider(scrapy.Spider):
    name = 'db'
    allowed_domains = ['movie.douban.com/']
    start_urls = ['http://movie.douban.com/']

    def parse(self, response):
        for page in range(40, 340, 20):
            #获取动态网页中获取异步加载每页真正请求地址，最大页数地址为320 ,共16页每页有20,17页有3
            all_urls = 'https://movie.douban.com/j/chart/top_list?type=5&interval_id=100%3A90&action=&start={}&limit=20'.format(page)
            yield scrapy.Request(url=all_urls, callback=self.sub_url, dont_filter=True)

    def sub_url(self, response):
                datas = json.loads(response.body)  # 将Json格式数据处理为字典类型

                # num = 323  # 获取所需爬取电影数量，最高323部电影
            #匹配获取到动态网页中的所有子网页链接
                    #首页，第一页,即获取二十部以内的电影信息数据
            # if num <= 20:
            #     for data in datas[0:num]:
            #         all_url_id = data['id']  # 爬取的链接id
            #         sub_urls = 'https://movie.douban.com/subject/' + all_url_id  #子网页链接
            #         yield scrapy.Request(url=sub_urls, callback=self.get_move_info, dont_filter=True)
            # else:
                #https://movie.douban.com/subject/1303408/comments?start=40&limit=20&status=P&sort=new_score

                #获取每部电影的链接ID号
                for data in datas[0:20]:  #每页最有20部，选取前20条
                    all_url_id = data['id']  # 爬取的链接id
                    for review in range(0, 520, 20):    # 二十部电影的子网页链接
                        sub_urls = 'https://movie.douban.com/subject/{}/comments?start={}&limit=20&status=P&sort=new_score'.format(all_url_id, review)
                        yield scrapy.Request(url=sub_urls, callback=self.get_url_info, dont_filter=True)



    def get_url_info(self, response):

            movice = re.sub('短评', '', response.xpath('//div[@id="content"]/h1/text()').get())
            #获得每一页评论评语 以及评论者用户的相关信息
            user_urls = response.xpath('//span[@class="comment-info"]//@href').getall()  # 评论者用户 个人空间网页地址
            if user_urls:
                user_name = response.xpath('//span[@class="comment-info"]/a/text()').getall()   # 评论者用户名
                user_star = response.xpath('//span[@class="comment-info"]/span[2]//@title').getall()# 评论者 给出的评分

                user_review_time = response.xpath('//span[@class="comment-time "]//@title').getall()   # 评论时间
                user_likes_number = response.xpath('//span[@class="votes vote-count"]/text()').getall()   # 评论被点赞数
                user_review = response.xpath('//span[@class="short"]/text()').getall()   # 评论者 评语


                for i in range(len(user_urls)):
                    if re.findall('\d', user_star[i]):
                        user_star[i] = None

                    #将电影评论页爬取的数据，传到下一函数中，最终导入MySQL中
                    yield scrapy.Request(url=user_urls[i], callback=self.get_user_info, meta={'user_url': user_urls[i],
                                                                                        'movice':''.join(movice),
                                                                                        'user_name': user_name[i],
                                                                                          'user_star': user_star[i],
                                                                                          'user_review_time': user_review_time[i],
                                                                                          'user_likes_number': user_likes_number[i],
                                                                                          'user_review': user_review[i]},
                                     dont_filter=True)

            else:
                pass



    def get_user_info(self, response):

        item = DoubanItem()
        item['movice'] = response.meta.get('movice')  #电影名

        item['user_name'] = response.meta.get('user_name')  # 评论者用户 个人空间网页地址
        item['user_star'] = response.meta.get('user_star')  # 评论者用户名
        item['user_review_time'] = response.meta.get('user_review_time')   # 评论时间
        item['user_likes_number'] = response.meta.get('user_likes_number')   # 评论被点赞数
        item['user_review'] = ''.join(re.findall('[\u4E00 -\u9FA5\\s]+', response.meta.get('user_review')))  # 评论者 评语


        item['user_place'] = response.xpath('//div[@class="user-info"]/a/text()').get()  # 评论者用户常居城市
        item['user_registration_time'] = response.xpath('//div[@class="pl"]/text()').getall()[1]  # 用户注册时间
        item['user_review_number'] = response.xpath('//div[@id="review"]//span[@class="pl"]/a/text()').get()  # 用户总共评论数量
        item['user_movice_watching'] = response.xpath('//div[@id="movie"]//span[@class="pl"]/a[1]/text()').get()  # 用户在看电影量
        item['user_movice_want'] = response.xpath('//div[@id="movie"]//span[@class="pl"]/a[2]/text()').get() # 用户在看电影量
        item['user_movice_watch'] = response.xpath('//div[@id="movie"]//span[@class="pl"]/a[3]/text()').get()  # 用户在看电影量
        item['user_follow'] = ''.join(re.findall('被(.*?)人关注', response.xpath('//p[@class="rev-link"]/a/text()').get()))  # 用户被关注量

        yield item

        # item = DoubanItem()
        # item['url'] = a
        # yield item