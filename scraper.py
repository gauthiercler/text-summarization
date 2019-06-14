import wikipediaapi


def get_article(topic):
    wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
    try:
        p_wiki = wiki.page(topic)
    except:
        print("Page " + topic + " Exists: False")
        quit()

    return p_wiki.text, p_wiki.summary
