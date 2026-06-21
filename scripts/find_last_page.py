# -*- coding: utf-8 -*-
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
import django; django.setup()
import requests
from dashboard.mofe_daily import LIST_URL, BBS_ID, MENU_NO, DEFAULT_HEADERS, fetch_list_page

s = requests.Session()
lo, hi = 1, 651
last_with = 0
while lo <= hi:
    mid = (lo + hi) // 2
    items = fetch_list_page(s, mid)
    n = len(items)
    print(f'page {mid}: {n} items', items[0].title if items else '')
    if n > 0:
        last_with = mid
        lo = mid + 1
    else:
        hi = mid - 1

print('LAST PAGE WITH ITEMS =', last_with)
items = fetch_list_page(s, last_with)
print('  oldest:', items[-1].title, items[-1].data_date)
print('  newest:', items[0].title, items[0].data_date)
