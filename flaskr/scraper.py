import wikipediaapi
import os

class Scraper:
    '''
    Scraper class will scrap wikipedia english page and save the text at the list.
    User can save all the data as txt file.
    '''
    
    def __init__(self):
        self.data = []
        
        
    def run(self, title):
        '''
        input
        
            title: wikipediaapi module scrap web page by only title of the page that user want to scrap
        '''
        # set wikipedia english web page
        wiki = wikipediaapi.Wikipedia( language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
        
        try:  
            p_wiki = wiki.page(title)
        except:
            print("Page " + title + " Exists: False")
            quit()
        self.data.append(p_wiki.text)
        
        
    def get_random_page(self, number_of_pages=1):
        '''
        This method is to scrap multiple random pages in wikipedia.
        
        Input
            number_of_pages: given integer number, scrap
        '''
        pass


    
    
        
    def savetxt(self):
        '''
        Save all the data in the list as txt file.
        '''
        
        for i, text in enumerate(self.data):
            path = './text' + str(i+1) + '.txt'
            try:
                f = open(path, 'w', encoding='utf8')
            except:
                print('cannot open' + path)
                quit()
                    
            f.write(text)
            f.close()
            

scraper = Scraper()
scraper.run('dfdac')
scraper.run('C++')
scraper.run('Java')
scraper.run('programming language')
scraper.savetxt()