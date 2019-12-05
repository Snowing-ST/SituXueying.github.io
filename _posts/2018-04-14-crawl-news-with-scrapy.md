---
title: 利用scrapy爬取网络新闻
author: Situ
layout: post
categories: [big data]
tags: [爬虫,scrapy]
---

<font face="仿宋" >用scrapy框架爬取在[新浪新闻](search.sina.com.cn)中用关键词搜索的结果。</font>

scrapy优势：
1. 新建爬虫项目时，只需改动原因框架一些内容，无需从头写代码
2. 执行程序时，在一定范围内爬取出错自动跳过，不会打断程序，无需定义出错识别和处理办法

### 安装crapy

打开anaconda prompt，输入```pip install scrapy```

### 新建scrapy项目

项目名称为```newsSpider```

#### 1. 打开anaconda prompt，cd到到指定文件夹，输入：```scrapy startproject newsSpider```，
#### 2. 再输入```tree newsSpider```可看到该文件夹中自动生成以下文件：
```
newsSpider/
    scrapy.cfg            # deploy configuration file
    newsSpider/             # project's Python module, you'll import your code from here
        __init__.py
        items.py          # project items definition file
        middlewares.py    # project middlewares file
        pipelines.py      # project pipelines file
        settings.py       # project settings file
        spiders/          # a directory where you'll later put your spiders
            __init__.py
```
#### 3. 最后cd到```./newsSpider/newsSpider/spiders```中，输入```scrapy genspider TraderWar search.sina.com.cn```
则生成```TraderWar.py```核心爬虫文件，其中有一行为```allowed_domains = ["search.sina.com.cn"]```，即指定了爬虫网站。

### 改写核心爬虫文件

```TraderWar.py``` 初始代码如下所示：
```python
import scrapy
class TraderwarSpider(scrapy.Spider):
    name = "TraderWar"
    allowed_domains = ["search.sina.com.cn"]
    start_urls = ['http://search.sina.com.cn/']

    def parse(self, response):
        pass
```

#### 1. 确定```start_urls```
在浏览器中打开search.sina.com.cn，输入搜索词，如“中美贸易战”，点击搜索结果的第2页，得到网址
http://search.sina.com.cn/?q=%D6%D0%C3%C0%C3%B3%D2%D7%D5%BD&c=news&from=index&col=&range=&source=&country=&size=&time=&a=&page=2&pf=0&ps=0&dpc=1
将page=2中的2替换成任一数字，则得到了每一页搜索的网址

#### 2. 提取每条新闻标题背后的链接
每一个搜索页面的网址将被传入到```parse```函数，在浏览器“检查”中查看网页源代码，找到每条新闻的标题链接的xpath路径，则可提取链接
#### 3. 指定页面内容中爬取的元素
提取的新闻链接传入```parse_detail```函数，scrapy将自动访问这些链接（即新闻详情页），在浏览器“检查”中查看网页源代码，找到新闻标题、发布时间、来源、全文内容等的xpath路径，提取这些元素保存至```NewsspiderItem```，这是一个在```items.py```中自定义的类（下文将会叙述）。

```python
import scrapy
from newsSpider.items import NewsspiderItem

class TradewarSpider(scrapy.Spider):
    name = "TradeWar"
#    allowed_domains = ["search.sina.com.cn"]
    start_urls = [
            'http://search.sina.com.cn/?q=%D6%D0%C3%C0%C3%B3%D2%D7%D5%BD&c=news&from=index&col=&range=&source=&country=&size=&time=&a=&page='+'%s&pf=2131425448&ps=2134309112&dpc=1' % p for p in list(range(1,20)) #设定只爬取前20页
            ]

    def parse(self, response):
        for href in response.xpath('//*[@id="result"]/div/div/h2/a/@href'): #提取页面中每条新闻的标题的链接
            full_url = response.urljoin(href.extract())#该链接是相对链接，要改成完整链接
            yield scrapy.Request(full_url, callback=self.parse_detail)
            
    def parse_detail(self, response):
        print(response.status)
        print(response.xpath('//h1[@class="main-title"]/text()').extract()[0]) #标题
        news = NewsspiderItem()
        news["url"] = response.url
        news["title"] = response.xpath('//h1[@class="main-title"]/text()').extract()[0]
        news["time"] = response.xpath('//*[@id="top_bar"]/div/div[2]/span/text()').extract()[0]
        news["origin"] = response.xpath('//*[@id="top_bar"]/div/div[2]/a/text()').extract()
        news["origin_url"] = response.xpath('//*[@id="top_bar"]/div/div[2]/a/@href').extract()[0]
        news["detail"] = "\n".join(response.xpath('//div[@class="article"]/div/p/text()').extract())+\
        "\n".join(response.xpath('//div[@class="article"]/p/text()').extract())+\
        "\n".join(response.xpath('//div[@class="article"]/div/div/text()'))
        yield news
```

#### 4. 其他文件的修改

在```items.py```自定义```NewsspiderItem```类，用于存放要爬取的内容

```python
class NewsspiderItem(scrapy.Item):
    title = scrapy.Field()          
    time = scrapy.Field()      
    origin = scrapy.Field()  
    origin_url = scrapy.Field() 
    detail = scrapy.Field()
    url = scrapy.Field()
#    abstract = scrapy.Field()
```

在```pipelines.py```中自定义```NewsspiderPipeline```，用于对爬取元素的简单处理。可定义多个pipelines，在setting中决定启用哪一个。

```python
class NewsspiderPipeline(object):
    def process_item(self, item, spider):
        item["time"] = item["time"][:11]#只需前11位
        item["detail"] = item["detail"].strip()#去掉空格
        return item
```

在```setting.py```中，关闭机器人协议

    通俗来说， robots.txt 是遵循 Robot协议 的一个文件，它保存在网站的服务器中，它的作用是，告诉搜索引擎爬虫，本网站哪些目录下的网页 不希望 你进行爬取收录。在Scrapy启动后，会在第一时间访问网站的 robots.txt 文件，然后决定该网站的爬取范围。

    当然，我们并不是在做搜索引擎，而且在某些情况下我们想要获取的内容恰恰是被 robots.txt 所禁止访问的。所以，某些时候，我们就要将此配置项设置为 False ，无需遵守Robot协议

```python
# Obey robots.txt rules
ROBOTSTXT_OBEY = False
```

选择使用哪几个pipelines。不需要则注释。
```python
ITEM_PIPELINES = {
    'newsSpider.pipelines.NewsspiderPipeline': 300,
#    'newsSpider.pipelines.NewsAbsspiderPipeline': 600,
}

```
#### 5. 运行```TraderWar.py``` 并保存爬虫结果

打开anaconda prompt，cd到newsSpider/newsSpider，运行```scrapy crawl TradeWar -o data.csv -s FEED_EXPORT_ENCODING=gbk```

或者，cd到newsSpider/newsSpider/spiders目录下，运行```scrapy runspider TradeWar.py -o data.csv -s FEED_EXPORT_ENCODING=gbk```
加上```FEED_EXPORT_ENCODING=gbk``` 可解决csv文件中文乱码问题。

爬取文件如下所示：
![news-crawl-result](http://localhost:4000/assets/images/post_images/news-crawl-result.png)


### 只在搜索页面爬取新闻摘要，无需全文内容
则核心爬虫文件改为：
```python
import scrapy
from newsSpider.items import NewsAbsspiderItem

class TradewarlistSpider(scrapy.Spider):
    name = "TradeWarList"
    allowed_domains = ["search.sina.com.cn"]
    start_urls = ["http://search.sina.com.cn/?q=%D6%D0%C3%C0%C3%B3%D2%D7%D5%BD&range=all&c=news&sort=rel"]

    def parse(self, response):
        # 请求第一页
        yield scrapy.Request(response.url, callback=self.parse_next)

#         请求其它页
        for page in response.xpath('//*[@id="_function_code_page"]/a')[:9]:
            link = response.urljoin(page.xpath('@href').extract()[0])
            print(link)
            yield scrapy.Request(link, callback=self.parse_next)

    def parse_next(self, response):
        for item in response.xpath('//*[@id="result"]/div/div'):
            news = NewsAbsspiderItem()
            print(item.xpath('h2/span/text()').extract()[0])
            news['title'] = item.xpath('h2/a')[0].xpath("string(.)").extract()[0]
            news['abstract'] = item.xpath('p')[0].xpath("string(.)").extract()[0]
            news['source'] = item.xpath('h2/span/text()').extract()[0]
            news['time2'] = item.xpath('h2/span/text()').extract()[0]
            news["url"] = item.xpath('h2/a/@href').extract()[0]
            yield news
```

其他代码文件修改类似。

完整代码欢迎访问我的[github](https://github.com/Snowing-ST/Statistical-Case-Studies/tree/master/Lab5%20Scraping%20with%20Scrapy)！


tips：本人尚未探索出scrapy又能在搜索页面爬取摘要，又能在新闻详情页爬取新闻全文的方法，如有建议，欢迎交流！若两者都想爬取，我用request+xpath可以做到，请参见下篇博客[利用xpath爬取网络新闻](https://snowing-st.github.io/big%20data/2018/04/20/crawl-news-with-xpath.html)。




