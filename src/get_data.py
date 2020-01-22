from bs4 import BeautifulSoup
import urllib.request
import requests
import shutil


def get_links():
    interested = []
    prefix="https://data.mendeley.com"
    target = "page.html"


    errors = ["C11P34E1"]
    counter = 0
    with open(target) as fp:
        soup = BeautifulSoup(fp)
        for a in soup.find_all(href=True):            
            if a["href"].endswith(errors[0]+".jpg?dl=1"):
                counter += 1            
                interested.append(prefix+a["href"])

    return interested


def download_links(interested):
    for id,url in enumerate(interested):
        print(id + 1)
        url = url[:-5]
        name = url.split('/')[-1]
        response = requests.get(url, stream=True)
        with open("../data/images/"+name, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        



if __name__ == "__main__":
    interested = get_links()
    download_links(interested)
