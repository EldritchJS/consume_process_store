import json
import time
import random
import re
import requests
import os
import urllib.request

from tqdm import tqdm
from bs4 import BeautifulSoup


def write_json(json_dict, file_path):
    with open(file_path, 'w+') as f:
        json.dump(json_dict, f)


def download_image(img_url, file_path):
    failed = True
    fail_count = 0

    while failed and fail_count < 5:
        try:
            urllib.request.urlretrieve(img_url, file_path)
            failed = False
        except:
            tqdm.write('Failed image download, sleeping then retrying')
            fail_count += 1
            time.sleep(random.randint(3, 7))


def scrape_post(user, post_id):
    post_url = f'https://t.me/{user}/{post_id}?embed=1'
    tqdm.write(f'scraping {post_url}')
    src = requests.get(post_url).text

    soup = BeautifulSoup(src, 'html.parser')

    if soup.find('div', class_='tgme_widget_message_error'):
        return None

    post_data = {}
    post_data['user'] = user
    post_data['post_id'] = post_id
    post_data['url'] = post_url

    try:
        post_data['title'] = soup.find('div', class_='tgme_widget_message_author accent_color').text
    except:
        post_data['title'] = ''

    if soup.find_all('a', {'style': re.compile('background-image:url')}):
        dat = soup.find_all('a', {'style': re.compile('background-image:url')})#[0]#['style']

        post_data['image_url'] = []
        # print(dat)
        for image_tag in dat:
            # print(image_tag)
            img = image_tag['style']
            ptr = re.search("http.*[)]", str(img))
            post_data['image_url'].append(str(img)[ptr.start():ptr.end() - 2])
    else:
        post_data['image_url'] = ''

    if soup.find('div', class_='tgme_widget_message_text js-message_text'):
        post_data['description'] = soup.find('div', class_='tgme_widget_message_text js-message_text').text
    else:
        post_data['description'] = ''

    try:
        post_data['views'] = soup.find('span', class_='tgme_widget_message_views').text
        post_data['date'] = soup.find('time', {'class':'datetime'})['datetime']
    except:
        return None

    tqdm.write('big succ')
    return post_data


def find_newest_post_num(user):
    post_url = f'https://t.me/s/{user}?embed=1'
    tqdm.write(f'Finding upper bound for {user}')
    src = requests.get(post_url).text

    soup = BeautifulSoup(src, 'html.parser')
    post_list = soup.find_all('a', {'href': re.compile(f'https://t.me/{user}', re.IGNORECASE)})

    try:
        newest_post = post_list[-1]['href'].split('/')[-1]
        print(f'Newest post found {newest_post}')
        return int(newest_post)
    except:
        print('Failed to find upper bound')
        return post_list
        # print(post_list[-1]['href'].split('/'))



def full_scrape(username_list):
    for user in tqdm(username_list, desc='users'):
        upper_bound = find_newest_post_num(user)

        if upper_bound == []:
            continue

        try:
            prev_scraped_dirs_max = max([int(f.name) for f in os.scandir(f'./{user}/') if f.is_dir()])
        except:
            prev_scraped_dirs_max = 1

        tqdm.write(f'Found {prev_scraped_dirs_max} previously scraped posts')

        for post_count in tqdm(range(prev_scraped_dirs_max + 1, upper_bound + 1), desc='Posts'):
            try:
                os.makedirs(f'./{user}/{post_count}')
                dat = scrape_post(user, post_count)
                time.sleep(random.randint(1, 3))
                if dat:
                    if dat['image_url'] != []:
                        img_count = 1
                        for img in dat['image_url']:
                            if img_count > 1:
                                post_count += 1
                                os.makedirs(f'./{user}/{post_count}')

                            download_image(img, f'./{user}/{post_count}/post_{post_count}_scraped_image_{img_count}.jpg')
                            img_count += 1
                            write_json(dat, f'./{user}/{post_count}/post_{post_count}_{dat["date"]}.json')

                    write_json(dat, f'./{user}/{post_count}/post_{post_count}_{dat["date"]}.json')
                else:
                    tqdm.write('No data, moving on')
            except FileExistsError:
                    tqdm.write(f'Folder {post_count} exists, assuming scraped previously')

        post_count = 1


if __name__ == '__main__':
    with open('./usernames.dat') as f:
        users = f.readlines()
    users = [user.rstrip() for user in users]
    users = list(set(users))

    full_scrape(users)
