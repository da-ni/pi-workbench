import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import argparse
import os

def scrape_wikipedia_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    content_text = soup.find('div', {'id': 'mw-content-text'})

    for script in content_text(["script", "style"]): # remove all javascript and stylesheet code
        script.extract()

    text = content_text.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


def scrape_all_linked_pages(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    content_text = soup.find('div', {'id': 'mw-content-text'})
    links = content_text.find_all('a', href=True)

    all_text = ""

    for link in links:
        full_url = urljoin(url, link['href'])
        if 'wiki' in full_url and '#' not in full_url: # filter out non-wiki and anchor links
            try:
                text = scrape_wikipedia_page(full_url)
                all_text += text + "\n\n"
            except:
                pass # ignore pages that can't be scraped for whatever reason

    return all_text


def main():
    parser = argparse.ArgumentParser(description='url of a wikipedia page you want to scrape and store the text:')
    parser.add_argument('--url', '-u', default="https://en.wikipedia.org/wiki/Pieter_Bruegel_the_Elder", type=str)
    parser.add_argument('--dir','-d', default='raw_data', type=str)
    parser.add_argument('--name','-n', default='bruegel.txt')
    parser.add_argument('--links','-l', default='no')
    
    args = parser.parse_args()
    
    if args.links == 'no':
        text = scrape_wikipedia_page(args.url)
    else:
        text = scrape_all_linked_pages(args.url)
    
    
    output_path = os.path.join(os.getcwd(),args.dir, args.name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    

if __name__ == "__main__": 
    main()