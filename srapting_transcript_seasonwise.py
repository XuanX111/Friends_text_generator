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

    def seasonNumber(self,lookup):
        f = open(os.path.join(self.response_folder, 'FriendsScript.txt'), 'r')

        for num, line in enumerate(f, 1):
            if lookup in line:

                print(lookup,'found at line:', num)
                break
        return num

    def cleaningData(self,start,end):

        f = open(os.path.join(self.response_folder, 'FriendsScript.txt'), 'r')
        diag_dict = dict()

        for i in range(6):
            diag_dict[self.main_character[i]] = []

        find_next = False

        # index = 0
        for i, line in enumerate(f): #maximum number of lines


            if i<start or i>=end:
                continue

            line = line.strip('\n')

            match = re.match(r'^([A-Z]\w+):', line)

            if match:

                name = match.group(0).lower()

                if name in self.main_character:
                    find_next = True
                    diag = line.split(':')[1:]
                    diag_dict[name].append(diag)
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
                   diag_dict[name_prev][-1].append(line)

            # index += 1
        return diag_dict
    def writeOutput(self,diag_dict,filename,lookup):

        diag_flat = {}
        for char in self.main_character:
            diag_flat[char] = {}
            diag_flat[char]['diag'] = [word for sublist in diag_dict[char] for word in sublist]
            diag_flat[char]['diag'] = ','.join(diag_flat[char]['diag'])
            diag_flat[char]['season'] = lookup

        # print(self.diag_flat['Rachel:'])
        df_diag = pd.DataFrame.from_dict(diag_flat, orient='index')
        # print(self.df_diag)


        with open(os.path.join(self.response_folder, f'{filename}.pkl'), 'wb') as fo:
            pickle.dump(df_diag, fo)

        return diag_flat

def main():
    diag = script()
    print('Done Scipting')
    start = 0
    for i in range(2,12,1):
        if i==11:
            end = 117926
        else:
            extract = "Chapter " + str(i)
            end = diag.seasonNumber(extract)

        diag_dict = diag.cleaningData(start,end)

        print('Done cleaning')
        diag.writeOutput(diag_dict,f"chapter{i-1}",i)
        print('Done')
        start = end
if __name__ == "__main__":
    main()
