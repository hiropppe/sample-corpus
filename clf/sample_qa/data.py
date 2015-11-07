#! /usr/bin/env python
# -*- coding: utf-8 -*-

import codecs, urllib2, sqlite3, os, textwrap, argparse, shutil, glob
import xml.etree.ElementTree as ET

from string import Template

import config

yurl = 'http://chiebukuro.yahooapis.jp/Chiebukuro/V1'

cat_url = Template(yurl + '/categoryTree?appid=' + config.appid + '&categoryid=$catid')
new_url = Template(yurl + '/getNewQuestionList?appid=' + config.appid + '&category_id=$catid&condition=$cond&start=$start&results=20')

qa_ddl = """
  create table if not exists qa (
    qid int primary key,
    catid int,
    content text
  )
"""

qa_conn = None

#fetch_conds = ['open']
fetch_conds = ['open', 'vote', 'solved']
pos_category_ids = [ 2078297371 ]

def open_db():
    global qa_conn

    if 0 < config.db_file.rfind('/'):
        mkdir(config.db_file[:config.db_file.rfind('/')], remove=False)
    qa_conn = sqlite3.connect(config.db_file, isolation_level=None)
    qa_conn.execute(qa_ddl)

def close_db():
    global qa_conn

    try:
        qa_conn.close()
    except:
        pass

def fetch(start_category, start_page, num_fetch_page, depth):
    catids = []
    fetch_target_categories(start_category, catids, depth)
    
    for catid in catids:
        num_fetch = fetch_news(catid, start=start_page, offset=num_fetch_page)

def fetch_target_categories(parent, rets, target_depth, depth=0):
    if rets is None:
        rets = []

    if depth == target_depth:
        rets.extend(get_categories(parent))
        return
    else:
        depth += 1
        for c in get_categories(parent):
            fetch_target_categories(c, rets, target_depth, depth)

def get_categories(parent=''):
    url = cat_url.substitute(catid=parent)
    print 'Fetching: category {}'.format(parent)
    try:
        rsp = urllib2.urlopen(url).read()
        tree = ET.fromstring(rsp)
        return [id.text for id in tree.findall('.//{urn:yahoo:jp:chiebukuro}Id')]
    except Exception as e:
        print e
        return []

def fetch_news(catid, start=1, offset=10):
    tmp_start = start
    num_fetch = 0
    for cond in fetch_conds: 
        while start <= tmp_start + offset:
            url = new_url.substitute(catid=catid, cond=cond, start=start) 
            try:
                tree = ET.fromstring(urllib2.urlopen(url).read())
                print 'Fetching: category={}, condition={}, page={}'.format(catid, cond, start)
                for rs in tree.findall('.//{urn:yahoo:jp:chiebukuro}Result'):
                    insert_or_replace(rs)
                    num_fetch += 1
            except Exception as e:
                print e
            start += 1
        start = tmp_start
    return num_fetch

def insert_or_replace(rs):
    global qa_conn
    
    qid = rs.findtext('{urn:yahoo:jp:chiebukuro}QuestionId')
    catid = rs.findtext('{urn:yahoo:jp:chiebukuro}CategoryId')
    content = rs.findtext('{urn:yahoo:jp:chiebukuro}Content')
    
    qa_conn.execute('INSERT OR REPLACE INTO qa VALUES (?, ?, ?)', (qid, catid, content))

def clf_gen():
    global qa_conn

    tmp_dir = os.path.join(config.corpus_dir, 'tmp')
    mkdir(tmp_dir)

    dump_qa_by_categories(tmp_dir)
    setup_binary_class_corpus(config.corpus_dir, tmp_dir)

def setup_binary_class_corpus(corpus_dir, tmp_dir):
    pos_dir = corpus_dir + '/1'
    neg_dir = corpus_dir + '/0'

    mkdir(pos_dir)
    mkdir(neg_dir)

    for pos_category in pos_category_ids:
        pos_category_dir = '/'.join([tmp_dir, `pos_category`])
        pos_files = glob.glob(pos_category_dir + '/*.txt')
        for pos_file in pos_files:
            shutil.copy(os.path.abspath(pos_file), pos_dir)

    pos_len = len(os.listdir(pos_dir))
    cat_len = len(os.listdir(tmp_dir))
    pos_cat_len = len(pos_category_ids)
    neg_cat_len = cat_len - pos_cat_len
    
    print 'num of positive:', pos_len
    print 'num of category:', cat_len
    print 'num of positive category:', pos_cat_len
    print 'num of negative category:', neg_cat_len

    neg_len_per_cat = max(1, pos_len/neg_cat_len) + 1
    print 'copy {} file from each negative category dir'.format(neg_len_per_cat) 

    for cat in glob.glob(tmp_dir + '/*'):
        if cat not in pos_category_ids:
            neg_cat_dir = os.path.abspath(cat)
            neg_files = glob.glob(neg_cat_dir + '/*.txt')[:neg_len_per_cat]
            for neg_file in neg_files:
                shutil.copy(os.path.abspath(neg_file), neg_dir)

def dump_qa_by_categories(tmp_dir):
    global qa_conn

    cur = qa_conn.cursor()
    try:
        cur.execute('select qid, catid, content from qa order by catid, qid')
        for row in cur:
            cat_dir = '/'.join([tmp_dir, `row[1]`])
            mkdir(cat_dir, remove=False)
            file = '/'.join([cat_dir, `row[0]` + '.txt'])
            with codecs.open(file, 'w', 'utf-8') as f:
                f.write(row[2])
            
    except Exception as e:
        print e

def mkdir(dir, remove=True):
    if remove:
        shutil.rmtree(dir, ignore_errors=True)
    
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''
            Yahoo!知恵袋生コーパス取得
            ''')
        )

    parser.add_argument('command', metavar='COMMAND', choices=['fetch', 'clfgen'], help='fetch | clfgen')
    parser.add_argument('-s', '--start', type=int, default=1, help='start page')
    parser.add_argument('-p', '--page', type=int, default=1, help='number of fetch page')
    parser.add_argument('-t', '--depth', type=int, default=2, help='category depth')
    parser.add_argument('-c', '--category', type=int, default=0, help='start category id')
    args = parser.parse_args()
    
    print 'Start to fetch Yahoo! 知恵袋データ'
    print ' Number of fetch page:', args.page
    print ' DB:', config.db_file

    try:
        open_db()
        
        if args.command == 'fetch':
            fetch(args.category, args.start, args.page, args.depth)
        elif args.command == 'clfgen':
            clf_gen()
    
    finally:
        close_db()
