import requests
import os
import json
from io import BytesIO
from bs4 import BeautifulSoup
from PIL import Image

IMG_PATH = "data/pictures/"
JSON_PATH = "data/json/"

# url = "doc.html"
url = "https://www.gettyimages.fr/photos/paysage?license=rf&family=creative&page=1&phrase=paysage&sort=best#license"
MAX_NB_PAGES = 99
NB_PAGES = 0

def urlpp(): # update url by replacing the page number, updates also NB_PAGES
    global NB_PAGES, url
    if (NB_PAGES > 0) and (NB_PAGES < MAX_NB_PAGES):
        url = url.replace("page=" + str(NB_PAGES), "page=" + str(NB_PAGES+1))
        NB_PAGES = NB_PAGES + 1
        return True
    if NB_PAGES == 0:
        NB_PAGES = NB_PAGES + 1
        return True
    return False

def get_html(url): # get html content from url
    resp = requests.get(url)
    return resp.content

def get_table_img(soup): # get all gi-assets division then return them as a list
    soup = soup.find("body").find_all("div", class_="content_wrapper")[0]
    soup = soup.section.find_all("div", class_="site-width")[0]
    soup = soup.main.section.find_all("div", class_="search-content__gallery-pager-wrapper")[0]
    soup = soup.find_all("div", class_="search-content__gallery")[0]
    soup = soup.find_all("div", class_="search-content__gallery-results")[0]
    soup = soup.find_all("div", class_="search-content__gallery-assets")[0]
    soup = soup.find_all("gi-asset")
    print("NB_ASSETS", len(soup))
    return soup

def get_img_urls(soup): # from all gi-assets division extracted from get_table_img() we get urls of images from those division
    res = []
    for s in soup:
        s = s.find_all("a")[0].find_all("figure")[0]
        s = s.find_all("img")[0]
        res.append(s["src"])
    return res


def save_urls(page, urls):
    with open(JSON_PATH + str(page) + ".json", "w+") as f:
        json.dump({page : urls}, f)

def load_url(file):
    key = None
    value = None

    with open(JSON_PATH + file, 'r') as f:
        data = json.loads(f.read())
        key = list(data.keys())[0]
        value = data[key]

    return key, value

def load_urls():
    res = {}

    for _,_,files in os.walk(JSON_PATH):
        for file in files:
            key, value = load_url(file)
            res[key] = value

    return res

def get_all_pages(): # get all urls from all pages from 1 to 99 and saves urls to data/json
    try:
        while (urlpp()):
            print("\tNB_PAGES: " + str(NB_PAGES))
            print("Now searching: " + url)
            html_doc = get_html(url)

            print('html_doc...')
            soup = BeautifulSoup(html_doc, "html.parser")

            print('parsing...')
            soup = get_table_img(soup)
            res = get_img_urls(soup)

            save_urls(NB_PAGES, res)
    except Exception as e:
        print(e)
        return False

    return True

def download_imgs(page, urls): # download images and put them in data folder
    try:
        os.mkdir(IMG_PATH + str(page))
    except:
        pass

    i=0
    for url in urls:
        print(url)
        resp = requests.get(url).content
        img = Image.open(BytesIO(resp))
        img.save(IMG_PATH + str(page) + '/' + str(i) + ".png")
        i += 1

if __name__ == "__main__":
    print("launching...")
    # get_all_pages() # saving all pages in json while crawling web pages
    res = load_urls() # load json files (from get_all_pages())
    # print(res)

    print("\n\nRESULT:\n\n") # dowload images from urls in res
    for k in res.keys():
        print("\tPAGE:" + str(k) + ", urls:" + str(res[k][0][::1]))
        download_imgs(k, res[k])
