import logging
import os
import re
import pickle
import pandas as pd
import requests
import threading
from bs4 import BeautifulSoup

class script():

    def __init__(self):

        self.response_folder = '~/Documents/Metis/project4'
        self.response_folder = os.path.expanduser(self.response_folder)
        self.main_character = ('phoebe:', 'rachel:', 'ross:', 'monica:', 'chandler:', 'joey:')

    def scrapeData(self):

        url="https://fangj.github.io/friends/"
        soup = BeautifulSoup(requests.get(url).text, "lxml")
        with open(os.path.join(self.response_folder, 'transcript_search_page.html'), 'wb') as fo:
            fo.write(soup.prettify().encode('ascii', 'ignore'))

        list_head = soup.find_all('li')
        web_lst = [None]*len(list_head)

        for i in range(len(list_head)):
            episode = list_head[i].a.string
            dict = {}
            dict['episode']=episode
            url = list_head[i].find('a', href=True).get('href')
            dict['abs_url'] = str(f'https://fangj.github.io/friends/{url}')
            web_lst[i] = dict

        for i in range(len(list_head)):

            url = web_lst[i]['abs_url']
            soup = BeautifulSoup(requests.get(url).text, "lxml")
            script = soup.find_all('p', align='left')

            for s in script:

                if s.text is None:
                    continue
                try:
                    character = s.b.string
                    n = len(character)
                    if character not in web_lst[i]:
                        web_lst[i][character] = []

                    web_lst[i][character].append(s.text[n:])

                except:
                    pass

    def cleaningData(self):

        f = open(os.path.join(self.response_folder, 'FriendsScript.txt'), 'r')
        self.diag_dict = dict()

        for i in range(6):
            self.diag_dict[self.main_character[i]] = []
        index = 0
        find_next = False

        while index < 117926: #maximum number of lines

            line = f.readline()
            line = line.strip('\n')

            match = re.match(r'^([A-Z]\w+):', line)

            if match:

                name = match.group(0).lower()

                if name in self.main_character:
                    find_next = True
                    diag = line.split(':')[1:]
                    self.diag_dict[name].append(diag)
                    name_prev = name
            else:

                if re.match('SEASON [A-z]*?', line):
                    find_next = False
                elif re.match('SCENE [0-9]*?', line):
                    find_next = False
                elif re.match('^[0-9]', line):
                    find_next = False
                elif line in ['', '\r']:
                    find_next = False
                elif find_next == True and re.match("[^[(]", line):
                   self.diag_dict[name_prev][-1].append(line)

            index += 1

    def writeOutput(self):
        self.diag_dict
        self.diag_flat = {}
        for char in self.main_character:
            self.diag_flat[char] = [word for sublist in self.diag_dict[char] for word in sublist]
            self.diag_flat[char] = ','.join(self.diag_flat[char])

        # print(self.diag_flat['Rachel:'])
        self.df_diag = pd.DataFrame.from_dict(self.diag_flat, orient='index')
        # print(self.df_diag)

        with open(os.path.join(self.response_folder, f'df_diag.pkl'), 'wb') as fo:
            pickle.dump(self.df_diag, fo)

def main():
    diag = script()
    print('Done Scipting')
    diag.cleaningData()
    print('Done cleaning')
    diag.writeOutput()
    print('Done')

if __name__ == "__main__":
    main()
