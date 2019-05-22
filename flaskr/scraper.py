import bs4 as bs
import urllib.request


class Scraper:

    def __init__(self):
        self.data = []

    def run(self, url):
        scraped_data = urllib.request.urlopen(url)
        article = scraped_data.read()
        parsed_article = bs.BeautifulSoup(article, 'lxml')
        self.data = parsed_article.find_all('p')
