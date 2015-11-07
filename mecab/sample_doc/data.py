#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import sys
import os
import re
import codecs
import glob
import shutil
import nltk
import pprint
import urllib2
import sqlite3
import ConfigParser
import argparse
import textwrap
import MeCab

from os.path import join, abspath, basename

from bs4 import BeautifulSoup
 
class SampleCorpus(object):
 
    cid_prefix = '1000000'
    dtl_prefix = 'dtl' + cid_prefix
    
    url_page_list  = 'http://potaru.com/service/page/?sortType=createDateDesc'
    url_dtl_prefix = 'http://potaru.com/' + dtl_prefix
    
    url_dtl = 'http://potaru.com/dtl{cid}/'
    
    doc_db_create_ddl = """
        create table if not exists doc (
            cid varchar(10)
        ,   docid int
        ,   fileid int
        ,   sents text
        )
    """

    r_sent       = re.compile(u'[^ 「」！？。]*[！？。]')
    r_ignore_lex = re.compile(r'.*\'.*')

    def __init__(self, corpus_dir, dict=None):
        self.corpus_dir = corpus_dir if corpus_dir[0] == '/' else abspath(corpus_dir) 
        
        self.corpus_db   = join(self.corpus_dir, 'corpus.db')
        self.corpus_prop = join(self.corpus_dir, 'corpus.info')
        
        self.corpus_mecab_dir       = join(self.corpus_dir, 'mecab')
        self.corpus_mecab_dict_dir  = join(self.corpus_mecab_dir, 'dict')
        self.corpus_mecab_learn_dir = join(self.corpus_mecab_dir, 'learn')
        
        self.bib_name = 'bib.csv'
        self.chaki_export_name = 'all.cabocha'
        
        mkdir(self.corpus_dir, remove=False)

        if dict is None:
            mecab_opt = '-Ochasen'
        else:
            mecab_opt = '-Ochasen -d ' + dict

        self.mecab = MeCab.Tagger(mecab_opt)

        self.segment_doc_size = 500

    def fetch(self, start=-1, offset=-1):
        """
        テキストの取得
        """
        if start  == -1: start  = self._get_start()
        if offset == -1: offset = self._get_offset()

        print '文書情報を取得します ID範囲 {}-{}'.format(start, offset)
        
        self.open_doc_db()
        try:
            self.update_page_in_range(start, offset)
        finally:
            self.close_doc_db()
        
        print 'Done'

    def fetch_one(self, cid):
        print '文書情報を取得します'

        from_id = to_id = int(cid[len(self.cid_prefix):])

        self.open_doc_db()
        try:
            self.update_page_in_range(from_id, to_id)
        finally:
            self.close_doc_db()

        print '文書情報を取得しました'

    def fetch_list(self, cid_list_path):
        print '文書を取得します'

        self.open_doc_db()
        
        try:
            with open(cid_list_path) as f:
                for cid in f:
                    from_id = to_id = int(cid[len(self.cid_prefix):])
                    self.update_page_in_range(from_id, to_id)
        finally:
            self.close_doc_db()

        print '文書情報を取得しました'

    def all_gen(self):
        self.pln_gen()
        self.tag_gen()
        self.cps_gen()

    def pln_gen(self):
        """
        平文コーパスの生成
        """
        print '平文コーパスを生成中 ...'
        print ' DB:', self.corpus_db
       
        self.mk_segment_dirs()

        self.open_doc_db()
        try:
            cur = self.doc_conn.cursor()
            cur.execute('select docid, cid, sents from doc order by docid')
            
            for row in cur:
                docid, cid, sents = row[0], row[1], row[2]
                dir  = self.segment_dir_from_docid(docid, 'pln')
                file = self._doc_name(docid, cid, 'txt')
                with codecs.open(join(dir, file), 'w', 'utf-8') as f:
                    f.write(sents + '\n')
        
        except Exception, detail:
            print 'Unexpected error has occured while writing corpus.', detail
            raise
        finally:
            self.close_doc_db()
        
        num_total = 0
        for segdir, segid, in sorted(self.segment_dirs()):
            num_seg = len(os.listdir(self.segment_dir(segid, 'pln')))
            print ' seg' + `segid`, num_seg
            num_total += num_seg 
        print ' total', num_total

        print '生成されました\n'
    
    def tag_gen(self):
        """
        タグ付きコーパスの生成
        """
        print 'タグ付きコーパスを生成中 ...'
        
        self.tag_doc_gen()
        self.tag_bib_gen()
        
        print '生成されました\n'

    def cps_gen(self):
        """
        Chakiエクスポートコーパスとタグ付きコーパスのマージ及び正解コーパスの抽出
        """
        print 'ChakiコーパスとMeCab追加辞書、再学習用コーパスを生成中 ...'

        self.clean_mecab_learning_data()

        for segdir, segid in sorted(self.segment_dirs()):
            print 'セグメント' + `segid` + 'に入ります'
            self._load_bib_from_chaki_corpus(segid)
            self._split_chaki_corpus(segid)
            self._update_corpus_to_latest(segid)
            self._bib_merge(segid)
            self._mecab_learn_gen(segid)
       
        self._mecab_dict_gen()
        
        print '生成されました\n'
    
    def tag_doc_gen(self):
        """
        タグ付きコーパスの文書ファイルの作成
        """
        print 'CaboChaファイルを生成'

        for segdir, segid in sorted(self.segment_dirs()):
            print 'セグメント' + `segid` + 'に入ります'

            files = glob.glob(join(segdir, 'pln', '*.txt'))
            for f in files:
                with codecs.open(f, 'r', 'utf-8') as fi:
                    plain_name = basename(f)
                    tagged_name = plain_name[0:plain_name.index('.')] + '.cabocha'
                    tagged_file = join(self.segment_dir(segid, 'tag'), tagged_name)
                    
                    with codecs.open(tagged_file, 'w', 'utf-8') as fo:
                        self._write_tagged_sents(fi, fo)

    def tag_bib_gen(self):
        """
        タグ付きコーパスの書誌情報ファイルの生成
        """
        print '書誌情報ファイルを生成'
        
        for segdir, segid in sorted(self.segment_dirs()):
            print 'セグメント' + `segid` + 'に入ります'
            
            tag_dir = self.segment_dir(segid, 'tag')
            bib_file = join(tag_dir, 'bib.csv')
            with codecs.open(bib_file, 'w', 'utf-8') as f:
                self.sent_id = 0
                f.write('sentence_id,text,document_id,document_attributes,sentence_attributes\r\n')
                self._foreach_doc_process_file(segid, f, self._bib_write)

    def clean_mecab_learning_data(self):
        shutil.rmtree(self.corpus_mecab_dir, ignore_errors=True)
        os.makedirs(self.corpus_mecab_dict_dir)
        os.makedirs(self.corpus_mecab_learn_dir)

    def _mecab_dict_gen(self):
        """
        MeCab追加辞書の出力
        正解データとMeCab解析結果の差分を語彙素として出力する
        """
        print '新語彙素を抽出 ...'
        sent = []
        lexset = set()
        lexset_mecab = set()
        with codecs.open(join(self.corpus_mecab_dict_dir, 'lex.csv'), 'w') as lexfile:
            files = glob.glob(join(self.corpus_mecab_learn_dir, '*.cabocha'))
            for f in files:
                with codecs.open(f, 'r', 'utf-8') as fi:
                    for lex in fi:
                        if self._is_lexline(lex):
                            lex = self._chaki2worklex(lex)
                            sent.append(lex[0:lex.index('\t')])
                            lexset.add(lex)
                        else:
                            encoded_sent = ''.join(sent)
                            node = self.mecab.parseToNode(encoded_sent)
                            node = node.next
                            while node:
                                if self.r_ignore_lex.match(node.surface) is None:
                                    mlex = self._mecab2worklex(node)
                                    lexset_mecab.add(mlex)
                                node = node.next
                            del sent[:]
                    
                    self._mecab_lex_write(lexfile, lexset, lexset_mecab)
                    
                    lexset.clear()
                    lexset_mecab.clear()

    def _mecab2worklex(self, node):
        """
        mecabの解析結果をchakiの語彙素表記との比較用の形式に変換する
        chakiのExportは活用のない語いついても基本形を出力するため基本形は見ないことにする
        """
        lst_feature = node.feature.split(',')
        lst_feature[6] = '*'
        if len(lst_feature) == 7:
            lst_feature.extend(['*', '*'])
        return node.surface + '\t' + ','.join(lst_feature)

    def _chaki2worklex(self, lex):
        """
        chakiの語彙素表記をmecabの解析結果との比較用の形式に変換する
            chaki -> 歩き\t動詞,自立,*,*,五段-カ行イ音便,連用形,歩く,アルキ,アルキ\tO 
            mecab -> 歩き\t動詞,自立,*,*,五段・カ行イ音便,連用形,歩く,アルキ,アルキ
        chakiのExportは活用のない語についても基本形を出力するため基本形は見ないことにする
        """
        encoded_lex = lex.encode('utf-8')
        lst_lex = re.split(r'\t|,', encoded_lex)
        lst_lex[5] = lst_lex[5].replace('-', '・')
        lst_lex[7] = '*'
        return lst_lex[0] + '\t' + ','.join(lst_lex[1:10])

    def _mecab_lex_write(self, file, learnset, originalset):
        for lex1 in learnset:
            if lex1 not in originalset:
                file.write(self._mecab_set_zero_cost(lex1) + '\n')

    def _mecab_set_zero_cost(self, lex):
        return lex.replace('\t', ',0,0,0,', 1)

    def _mecab_learn_gen(self, segid):
        """
        MeCab再学習用データの出力
        """
        for k in sorted(self.bibmap.keys()):
            bib = self.bibmap[k]
            if self._bib_status(bib) == '9':
                src = join(self.segment_dir(segid, 'cps'), self._cabocha_name_from_bib(bib))
                dst = self.corpus_mecab_learn_dir
                shutil.copy(src, dst)

    def _load_bib_from_chaki_corpus(self, segid):
        """
        Chaki出力コーパスの書誌情報の読み込み
        """
        self.bibmap = {}

        if not os.path.exists(self.chaki_export_file(segid)):
            return
            
        with codecs.open(self.chaki_export_file(segid)) as f:
            for line in f:
                if self._is_bibline(line):
                    bib = self._bib_from_bibline(line)
                    self.bibmap[bib[0]] = bib
                else:
                    break
    
    def _split_chaki_corpus(self, segid):
        """
        Chaki出力コーパスの文書単位へのファイル分割
        """
        
        if not os.path.exists(self.chaki_export_file(segid)):
            return

        with codecs.open(self.chaki_export_file(segid), 'r', 'utf-8') as f:
            split_file = None
            for line in f:
                if self._is_new_docline(line):
                    if split_file:
                        split_file.close()
                    split_file = self._open_docfile_exported(segid, line)
                elif split_file is not None and not self._is_tagline(line):
                    split_file.write(line.replace('\r\n', '\n'))
                else:
                    pass
            split_file.close()

    def chaki_export_file(self, segid):
        return join(self.segment_dir(segid, 'cki'), self.chaki_export_name)

    def _update_corpus_to_latest(self, segid):
        """
        Chaki出力コーパスの書誌情報を参照して、タグ付きコーパスを最新に更新する
        """
        
        dest_dir = self.segment_dir(segid, 'cps')
        src_dir = self.segment_dir(segid, 'tag')
        src_files = glob.glob(join(src_dir, '*.cabocha'))
        for f in src_files:
            name = basename(f)
            #doc_id = int(name[4:name.rindex('-')])
            doc_id = self._docid_from_filename(name) % self.segment_doc_size
            if doc_id in self.bibmap:
                bib = self.bibmap[doc_id]
                src = join(self.segment_dir(segid, 'cki'), self._cabocha_name_from_bib(bib))
            else:
                self.bibmap[doc_id] = self._bib_from_filename(name)
                src = abspath(f)
            shutil.copy(src, dest_dir)
    
    def _bib_merge(self, segid):
        cps_dir = self.segment_dir(segid, 'cps')
        bib_file = join(cps_dir, self.bib_name)
        with codecs.open(bib_file, 'w', 'utf-8') as f_bib:
            sent_id = 0
            f_bib.write('sentence_id,text,document_id,document_attributes,sentence_attributes\r\n')
            for k in sorted(self.bibmap.keys()):
                bib = self.bibmap[k]
                cabocha_file = join(cps_dir, self._cabocha_name_from_bib(bib))
                with codecs.open(cabocha_file, 'r', 'utf-8') as f_cab:
                    for i, sent in enumerate(self._get_sents_from_cabocha(f_cab, lambda s: ''.join(s)[0:20])):
                        if i == 0:
                            f_bib.write(self._bib_doc_from_bib(sent_id, sent, bib))
                        else:
                            f_bib.write(self._bib_sent_from_bib(sent_id, sent, bib))
                        sent_id += 1

    def _get_sents_from_cabocha(self, file, edit=lambda s:''.join(s)):
        sents = []
        sent = []
        for lex in file:
            if self._is_lexline(lex):
                sent.append(lex[0:lex.index('\t')])
            else:
                if sent:
                    sents.append(edit(sent))
                del sent[:]
        return sents

    def mk_segment_dirs(self):
        self.open_doc_db()
        try:
            cur = self.doc_conn.cursor()
            cur.execute('select max(docid) from doc')
            ret = cur.fetchone()
            if not ret is None:
                segsize = int(ret[0])/self.segment_doc_size + 1
                for i in range(0, segsize):
                   self.mk_segment_dir(i) 
        finally:
            self.close_doc_db()

    def mk_segment_dir(self, segid):
        for n in ['pln', 'tag', 'cki', 'cps']:
            dir = self.segment_dir(segid, n)
            if not os.path.exists(dir):
                os.makedirs(dir)

    def segment_id(self, docid):
        return int(docid)/self.segment_doc_size

    def segment_id_from_dirname(self, dir_name):
        return int(dir_name[len('seg')])

    def segment_dir(self, segid, name):
        return join(self.corpus_dir, 'seg' + `segid`, name)

    def segment_dir_from_docid(self, docid, name):
        return self.segment_dir(self.segment_id(docid), name)

    def segment_dirs(self):
        return [(d, self.segment_id_from_dirname(basename(d))) for d in glob.glob(join(self.corpus_dir, 'seg*'))]
    
    def _is_lexline(self, line):
        return not line.startswith('* ') and re.match(r'^EOS\s*$', line) is None

    def _is_bibline(self, line):
        return line.startswith('#! DOCID')

    def _bib_docid(self, bib):
        return bib[0]

    def _bib_filepath(self, bib):
        return bib[1]

    def _bib_bibid(self, bib):
        return bib[2]

    def _bib_status(self, bib):
        return bib[3]

    def _bib_from_filename(self, filename):
        r = re.compile('doc\-([0-9]+)\-([0-9]+)\..+$')
        m = r.search(filename)
        global_doc_id = int(m.group(1))
        doc_id = global_doc_id % self.segment_doc_size 
        return (doc_id, 'potaru/cps/' + filename, m.group(2), '0', global_doc_id)

    def _bib_from_bibline(self, bibline):
        doc_id = int(re.search(r'DOCID\t([0-9]+)\t', bibline).group(1))
        file_path = re.search(r'<FilePath>(.+)</FilePath>', bibline).group(1)
        bib_id = re.search(r'<Bib_ID>(.+)</Bib_ID>', bibline).group(1)
        status = re.search(r'<Status>([0-9]+)</Status>', bibline).group(1)
        global_doc_id = self._docid_from_filename(file_path)
        return (doc_id, file_path, bib_id, status, global_doc_id)

    def _docid_from_filename(self, filename):
        r = re.compile('doc\-([0-9]+)\-([0-9]+)\..+$')
        m = r.search(filename)
        return int(m.group(1))

    def _is_new_docline(self, line):
        return line.startswith('#! DOC ')

    def _is_tagline(self, line):
        return line.startswith(('#',))

    def _open_docfile_exported(self, segid, line):
        doc_id = int(line[len('#! DOC '):].strip())
        bib = self.bibmap[doc_id]
        doc_file = join(self.segment_dir(segid, 'cki'), self._cabocha_name_from_bib(bib))
        return codecs.open(doc_file, 'w', 'utf-8')

    def _bib_write(self, file, row):
        for i, sent in enumerate(self.get_sents(row[2])):
            if i == 0:
                file.write(self._bib_doc_from_row(sent, row))
            else:
                file.write(self._bib_sent_from_row(sent, row))
            self.sent_id += 1

    def _bib_doc_from_bib(self, sent_id, sent, bib):
        return `sent_id` + ',"' + sent[0:20] + '",' + `bib[0] % self.segment_doc_size` + ',"<Bib_ID>' + bib[2] + '</Bib_ID><FilePath>' + self._cabocha_name_from_bib(bib) + '</FilePath><Status>' + bib[3] + '</Status>",""\r\n'

    def _bib_sent_from_bib(self, sent_id, sent, bib):
        return `sent_id` + ',"' + sent[0:20] + '",' + `bib[0] % self.segment_doc_size` + ',"",""\r\n' 

    def _bib_doc_from_row(self, sent, row):
        return `self.sent_id` + ',"' + sent[0:20] + '",' + `row[0] % self.segment_doc_size` + ',"<Bib_ID>' + row[1]+ '</Bib_ID><FilePath>' + self._cabocha_name_from_row(row) + '</FilePath><Status>0</Status>",""\r\n'

    def _bib_sent_from_row(self, sent, row):
        return `self.sent_id` + ',"' + sent[0:20] + '",' + `row[0] % self.segment_doc_size` + ',"",""\r\n'

    def _cabocha_name_from_bib(self, bib):
        return self._doc_name(bib[4], bib[2], 'cabocha')

    def _cabocha_name_from_row(self, row):
        return self._doc_name(row[0], row[1], 'cabocha')

    def _doc_name(self, docid, cid, qualifier):
        return 'doc-' + '{0:05d}'.format(docid) + '-' + cid + '.' + qualifier
    
    def _foreach_doc_process_file(self, segid, file, func):
        self.open_doc_db()
        cur = self.doc_conn.cursor()
        try:
            cur.execute("""
                        select docid, cid, sents
                        from doc
                        where docid/? = ? order by docid
                        """, (self.segment_doc_size, segid))
            for row in cur:
                func(file, row)
        except Exception, detail:
            print 'Unexpected error: ', detail
            raise
        finally:
            self.close_doc_db()

    def update_page_in_range(self, start, offset):
        for sents in self.fetch_page(start, offset):
            # pass empty sentence
            if sents[1]:
                self.update_page(sents)
                #self._mark_update_db_done(sents[2])
            else:
                print 'Could not extract any sentence. ', sents

    def update_page(self, sents):
        cur = self.doc_conn.cursor()
        try:
            cid = sents[0]
            cur.execute(u'select docid, fileid from doc where cid = ?', (cid,))
            ret = cur.fetchone()
            
            if ret is None:
                self.doc_conn.execute(u'insert into doc (cid, docid, fileid, sents) values (?, ?, ?, ?)',
                                    (cid, self.next_docid, self.next_docid, '\n'.join(sents[1])))
                print sents[0] + ' is added: DOC ' + `self.next_docid`
                # increment docid
                self.next_docid += 1
            else:
                # no update
                self.doc_conn.execute('update doc set sents = ? where cid = ?', ('\n'.join(sents[1]), cid))
                print sents[0] + ' is updated: DOC ' + `ret[0]`
        except sqlite3.Error as e:
            print e

    def _mark_update_db_done(self, seq):
        cfg = ConfigParser.SafeConfigParser()
        cfg.read(self.corpus_prop)
        current = cfg.getint('page', 'next.page')
        if(current <= seq + 1):
            cfg.set('page', 'next.page', `seq + 1`)
            with open(self.corpus_prop, 'w') as f:
                cfg.write(f)
   
    def _write_tagged_sents(self, fi, fo):
        for sent in fi:
            for word in self._get_tagged_sent(sent):
                fo.write(word + '\n')

    def _get_tagged_sent(self, sent):
        encoded_text = sent.encode('utf-8')
        node = self.mecab.parseToNode(encoded_text)
        words = ['* 0 -1D 0/0 0']
        node = node.next
        while node:
            if self.r_ignore_lex.match(node.surface) is None:
                words.append(''.join([node.surface, '\t', node.feature]).decode('utf-8'))
            node = node.next
        words = words[0:len(words)-1]
        words.append('EOS')
        return words
    
    def _get_start(self):
        cfg = ConfigParser.SafeConfigParser()
        cfg.read(self.corpus_prop)
        if cfg.has_option('page', 'next.page'):
            return cfg.getint('page', 'next.page')
        else:
            cfg.add_section('page')
            cfg.set('page', 'next.page', '10000')
            with open(self.corpus_prop, 'w') as f:
                cfg.write(f)
            return 10000
 
    def _get_offset(self):
        html = urllib2.urlopen(self.url_page_list).read()
        soup = BeautifulSoup(html)
        href = soup.select('a[href^="' + self.url_dtl_prefix + '"]')[0]['href']
        return int(re.search(self.dtl_prefix + '([0-9]+)', href).group(1))
    
    def _write_bib(self, cur, f, fileid):
        print ' Set bib info ...'
        cur.execute(u'select cid, docid, sents from doc where fileid = ? order by docid', (fileid,))
        for r in cur:
            f.write('#! ' + 'DOCID\t' + `r[1]` + '\t<Bib_ID>' + r[0] + '</Bib_ID>\n')
 
    def _write_doc(self, cur, f, fileid):
        print ' Append doc ...'
        cur.execute(u'select cid, docid, sents from doc where fileid = ? order by docid', (fileid,))
        for r in cur:
            f.write('#! ' + 'DOC ' + `r[1]` + '\n')
            f.write(r[2] + '\n')
 
    def open_doc_db(self):
        self.doc_conn = sqlite3.connect(self.corpus_db, isolation_level=None)
        self.doc_conn.execute(self.doc_db_create_ddl)
        self._init_next_docid()
 
    def _init_next_docid(self):
        cur = self.doc_conn.cursor()
        cur.execute(u'select max(docid) + 1 from doc')
        next = cur.fetchone()
        if next[0] is None:
            self.next_docid = 0
        else:
            self.next_docid = next[0]
 
    def close_doc_db(self):
        try:
            self.doc_conn.close()
        except:
            pass
 
    def fetch_page(self, start, offset):
        for i in range(start, offset + 1):
            cid = self.cid_prefix + `i`
            url = self.url_dtl.format(cid=cid)
            try:
                print 'Parse {}'.format(url)
                yield (cid, self.get_sents(self.get_text(url)), i)
            except Exception, detail:
                print "Failed to retrive sentences: {}".format(url), detail
 
    def get_text(self, url):
        """Get text in potaru page"""
        try:
            html = urllib2.urlopen(url).read()
            soup = BeautifulSoup(html)
            content = soup.find(id='pageItems')
            return content.get_text()
        except Exception, detail:
            print "Error while get text.", detail
            raise
    
    def get_sents(self, text):
        """Get sentences from text"""
        #jp_sent_tokenizer = nltk.RegexpTokenizer(u'[^　！？。]+(！|？|。|$)')
        #noblankline_text = [l.strip() for l in text.split('\n') if re.match(r'^\s*$', l) == None]
        #sents = []
        #for line in noblankline_text:
        #    sents.extend(jp_sent_tokenizer.tokenize(line))
        
        text = ''.join([l.strip() for l in text.splitlines() if l.strip()])
        jp_sent_tokenizer = nltk.RegexpTokenizer(u'[^　「」！？。]+[！？。]')
        sents = jp_sent_tokenizer.tokenize(text)
        
        return sents
 
    def pp(self, obj):
        pp = pprint.PrettyPrinter(indent=4, width=160)
        str = pp.pformat(obj)
        return re.sub(r'\\u([0-9a-f]{4})', lambda x: unichr(int('0x'+x.group(1), 16)), str) 

def mkdir(dir, remove=True):
    if remove:
        shutil.rmtree(dir, ignore_errors=True)
    
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''
            MeCab再学習用のサンプルコーパスの生成ツール
            
            例:
                python data.py fetch data                 # 文書の取得
                python data.py -c 100000011379 fetch data # 文書の取得(ID指定)
                python data.py -l ./fetch.lst fetch data  # 文書の取得(リスト指定)
                python data.py allgen data  # コーパス生成
                python data.py plngen data  # 平文コーパスの生成
                python data.py -d ./dict/mecab-ipadic-2.7.0-20070801-latest taggen data # タグ付きコーパスの生成
                python data.py -d ./dict/mecab-ipadic-2.7.0-20070801-latest cpsgen data # Chakiコーパスの更新
            ''')
        )
    parser.add_argument('command', metavar='COMMAND',
                        choices=['fetch', 'allgen', 'plngen', 'taggen', 'cpsgen'],
                        help='fetch | plngen | taggen | cpsgen')
    parser.add_argument('corpus', metavar='CORPUS_DIRECTORY',
                        help='コーパスディレクトリ')
    parser.add_argument('-c', '--cid', metavar='CID', type=str, default=None)
    parser.add_argument('-l', '--cid_list', metavar='CID_LIST_FILE', type=str, default=None)
    parser.add_argument('-d', '--dict', metavar='MeCab_SYSTEM_DICTIONARY', type=str, default=None,
                        help='解析に使用するMeCabシステム辞書')
    args = parser.parse_args()

    cps = SampleCorpus(args.corpus, args.dict)

    if args.command == 'fetch':
        if not args.cid is None:
            cps.fetch_one(args.cid)
        elif not args.cid_list is None:
            cps.fetch_list(args.cid_list)
        else:
            cps.fetch()
    elif args.command == 'allgen':
        cps.all_gen()
    elif args.command == 'plngen':
        cps.pln_gen()
    elif args.command == 'taggen':
        cps.tag_gen()
    elif args.command == 'cpsgen':
        cps.cps_gen()
