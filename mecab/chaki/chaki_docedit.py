#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import codecs
import argparse
import textwrap
from os.path import join

import MySQLdb as my
import csv

corpus_db = None

def open_db(host, db, user, passwd, charset='utf8'):
    global corpus_db
    corpus_db = my.connect(host=host, db=db, user=user, passwd=passwd, charset='utf8')

def close_db():
    global corpus_db
    try:
        corpus_db.close()
    except:
        pass

q_sel_doc = 'select document_id from documenttag where tag = %s and description = %s'
q_sel_tag = 'select id from documenttag where document_id = %s and tag = %s'
q_upd_tag = 'update documenttag set description = %s where id = %s'
q_ins_tag = """
            insert into documenttag
              (id, tag, description, document_id)
            select
              (select max(id)+1 from documenttag) as id, %s, %s, %s
            """

def edit_doctag(csv_file):
    cur = corpus_db.cursor()

    f = open(csv_file, 'r')
    reader = csv.reader(f)
    for row in reader:
        bib_id = row[0]
        tag = row[1]
        value = row[2]

        cur.execute(q_sel_doc, ('Bib_ID', bib_id))
        doc_id_row = cur.fetchone()
        if doc_id_row is None:
            continue
        doc_id = doc_id_row[0]

        cur.execute(q_sel_tag, (doc_id, tag))
        target_tag_row = cur.fetchone()
        if target_tag_row is None:
            print 'Insert', (tag, value, doc_id)
            cur.execute(q_ins_tag, (tag, value, doc_id))
        else:
            doc_tag_id = target_tag_row[0]
            print 'Update', (doc_tag_id, value)
            cur.execute(q_upd_tag, (value, doc_tag_id))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''
        Usage: python script/chaki_docedit.py doctag.csv 
        ''')
        )
    parser.add_argument('document_tag_csv')
    parser.add_argument('-d', '--db', type=str, required=True)
    parser.add_argument('-c', '--host', type=str, default='localhost')
    parser.add_argument('-u', '--user', type=str, default='mysql')
    parser.add_argument('-p', '--password', type=str, default='')
    args = parser.parse_args()

    csv_file = args.document_tag_csv
    db = args.db
    host = args.host
    user = args.user
    pwd = args.password

    open_db(host, db, user, pwd)
    try:
        edit_doctag(csv_file)
    except Exception, detail:
        print detail
        raise
    finally:
        close_db()

