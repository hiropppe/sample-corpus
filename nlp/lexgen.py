#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types, sys, glob, codecs, gzip, math, nltk, sqlite3, os, re, argparse, textwrap
from collections import defaultdict
from multiprocessing import Pool

import MeCab
mecab = MeCab.Tagger('-Ochasen') 

numdoc = 0
term_freq = defaultdict(int)
bigram_freq = defaultdict(lambda: defaultdict(int))
doc_freq = defaultdict(int)
result = {}

accept_feature = [
      '接頭詞,数接続,*,*'
#    , '接頭詞,名詞接続,*,*'
    , '名詞,サ変接続,*,*'
    , '名詞,ナイ形容詞語幹,*,*'
    , '名詞,形容動詞語幹,*,*'
    , '名詞,動詞非自立的,*,*'
    , '名詞,一般,*,*'
    , '名詞,数,*,*'
    , '名詞,接続詞的,*,*'
    , '名詞,固有名詞,*,*'
    , '名詞,固有名詞,一般,*'
    , '名詞,固有名詞,人名,*'
    , '名詞,固有名詞,人名,一般'
    , '名詞,固有名詞,人名,姓'
    , '名詞,固有名詞,人名,名'
    , '名詞,固有名詞,組織,*'
    , '名詞,固有名詞,地域,*'
    , '名詞,固有名詞,地域,一般'
    , '名詞,固有名詞,地域,国'
    , '名詞,組織,*,*'
    , '名詞,地域,*,*'
    , '名詞,地域,一般,*'
    , '名詞,地域,国,*'
    , '名詞,接尾,*,*'
    , '名詞,接尾,一般,*'
    , '名詞,接尾,形容動詞語幹,*'
    , '名詞,接尾,助数詞,*'
    , '名詞,接尾,助動詞語幹,*'
    , '名詞,接尾,地域,*'
    , '名詞,接尾,特殊,*'
    , '名詞,代名詞,*,*'
    , '名詞,代名詞,一般,*'
    , '名詞,代名詞,縮約,*'
    , '名詞,非自立,*,*'
    , '名詞,非自立,一般,*'
    , '名詞,非自立,形容動詞語幹,*'
    , '名詞,非自立,助動詞語幹,*'
    , '未知語,*,*,*'
]

except_feature_patterns = [
      r'^((\_名詞,数,\*)+(\_名詞,接尾,(\*|一般|形容動詞語幹|助数詞|助動詞語幹|特殊))*)+\_$'
]

except_lr_features = [
#      '名詞,サ変接続,*,*'
#    , '名詞,ナイ形容詞語幹,*,*'
#    , '名詞,形容動詞語幹,*,*'
#    , '名詞,動詞非自立的,*,*'
      '名詞,数,*,*'
    , '名詞,接続詞的,*,*'
    , '名詞,接尾,*,*'
    , '名詞,接尾,サ変接続,*'
    , '名詞,接尾,一般,*'
    , '名詞,接尾,形容動詞語幹,*'
    , '名詞,接尾,助数詞,*'
    , '名詞,接尾,助動詞語幹,*'
    , '名詞,接尾,地域,*'
    , '名詞,接尾,特殊,*'
#    , '名詞,代名詞,*,*'
#    , '名詞,代名詞,一般,*'
#    , '名詞,代名詞,縮約,*'
#    , '名詞,非自立,*,*'
#    , '名詞,非自立,一般,*'
#    , '名詞,非自立,形容動詞語幹,*'
#    , '名詞,非自立,助動詞語幹,*'
]

except_lr_surfaces = ['～', '-', '.', ',', '(', ')', '人', '選手', '情報', '月']

term_ddl = """
    create table if not exists term (
        term varchar(50)
    ,   freq int
    ,   primary key(`term`)
    )
"""

idx_termfreq_ddl = 'create index if not exists idx_term_freq on term (`freq`)'

bigram_ddl = """
    create table if not exists bigram (
        term1 varchar(10)
    ,   term2 varchar(10)
    ,   freq int
    ,   primary key(`term1`, `term2`)
    )
"""

idx_bigramfreq_ddl = 'create index if not exists idx_bigram_freq on bigram (`freq`)' 

docfreq_ddl = """
    create table if not exists termdoc (
        term varchar(50)
    ,   freq int
    ,   primary key(`term`)
    )
"""

idx_docfreq_ddl = 'create index if not exists idx_termdoc_freq on termdoc (`freq`)'

stats_ddl = """
    create table if not exists stats (
        numdoc int
    )
"""

dic_dir = None

db = None
db_file = None
db_flush_threshould = 10000

## test

def test():
    term_freq.clear()
    bigram_freq.clear()
    doc_freq.clear()
    
    open_db()
    clean_db()
    for l in ['自然言語処理において、形態素解析は基礎となるタスクです。', 
            '形態素解析の次のタスクは、構文解析です。構文解析では、主に係り受け解析が行われ、文節間の依存関係が解析されます。'
            '形態素解析器の代表的なものとして、和布蕪や茶筅、案山子と行ったものがあります。'
            '係り受け解析器の代表的なものとして、南瓜、KNPがあります'
            '自然言語処理において、文書の特徴量は、bag of wordsと呼ばれるベクトル表現で近似的に表現されます。'
            '単語文書行列は、一般的に疎行列となるため、機械学習等の計算処理を適用する前に、LSI、LDAを使用して次元削減を行います。'
            '機械学習には、教師なし学習、教師あり学習があり、教師なしは学習データを用意する必要がないため、比較的簡単に開始できます。代表的な、教師なし学習として、クラスタリングがあります。'
            '教師ありは、学習データを用意する必要がありますが、その可能性は強力であり、画像認識や音声認識、自然言語処理の分野で広く活躍しています。']:
        analyze_doc(l)

    flush_freq(force=True)
    
    print 'Bigram Stats'
    print_noun_stats(['自然', '言語', '処理', '形態素', '解析', '構文', '係り', '受け', '和布', '蕪', '案山子'
                    , '単語', '文書', '行列', 'LSI', 'LDA', '機械', '学習', '教師', '画像', '音声', '認識', '一般', '代表', '的'])
    print 'Noun Score'
    print_noun_score(['自然_言語_処理', '形態素_解析', '構文_解析', '係り_受け_解析', 
                    '単語_文書_行列', '機械_学習', '教師_なし_学習', '教師_あり_学習', 
                    '画像_認識', '音声_認識', '代表_的', '一般_的', '近似_的', '解析', '処理', '単語', '文書', '行列', 
                    '機械', '学習', '教師', '代表', '的'])

    close_db()

def test_score():
    # 語頻度
    print 'Populate TF ...'
    run_tf()
    write('tf', u'./TF.tsv')

    print 'Populate IDF ...'
    run_idf()
    write('idf', u'./IDF.tsv') 

    print 'Populate TF-IDF'
    run_tfidf()
    write('tfidf', u'./TF-IDF.tsv') 

    # 連結種類LR
    print 'Populate Pattern LR ...'
    run_lr(f=mlr, fl=ldn, fr=rdn, rid='dnlr')
    write('dnlr', u'./連接種類LR法.tsv')
    
    print ' TF Boosting ...'
    #run_lr(fl=ldn, fr=rdn, fw=tf)
    run_cached_lr('tf_dnlr', 'dnlr', 'tf')
    write('tf_dnlr', u'./TF連接種類LR法.tsv')

    print ' TF-IDF Boosting ...'
    #run_lr(fl=ldn, fr=rdn, fw=tfidf)
    run_cached_lr('tfidf_dnlr', 'dnlr', 'tfidf')
    write('tfidf_dnlr', u'./TFIDF連接種類LR法.tsv')

    # 連結頻度LR
    print 'Populate Freq LR ...'
    run_lr(f=mlr, fl=ln, fr=rn, rid='nlr')
    write('nlr', u'./連接頻度LR法.tsv')

    print ' TF Boosting ...'
    #run_lr(fl=ln, fr=rn, fw=tf)
    run_cached_lr('tf_nlr', 'nlr', 'tf')
    write('tf_nlr', u'./TF連接頻度LR法.tsv')

    print ' TF-IDF Boosting ...'
    #run_lr(fl=ln, fr=rn, fw=tfidf)
    run_cached_lr('tfidf_nlr', 'nlr', 'tfidf')
    write('tfidf_nlr', u'./TFIDF連接頻度LR法.tsv')

    # MC-value
    #print 'Populate MC-value'
    #run_mc_value()
    #write('mcv', './MC-value.tsv')
    #print 'done'

def print_noun_stats(nouns):
    print '{:15}\t{}\t{}\t{}\t{}'.format('', 'LDN', 'RDN', 'LN', 'RN')
    for n in nouns:
        print '{:15}\t{}\t{}\t{}\t{}'.format(n, ldn(n), rdn(n), ln(n), rn(n))

def print_noun_score(nouns):
    print '{:15}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
        '', 'TF', 'TFIDF',
        '種類LR', '頻度LR',
        '種類FLR(tf)', '頻度FLR(tf)',
        '種類MLR', '頻度MLR',
        '種類FMLR(tf)', '頻度FMLR(tf)',
        'MC-value')
    for n in nouns:
        print '{:15}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}' .format(
            n, tf(n), tfidf(n),
            lr(n, ldn, rdn), lr(n, ln, rn),
            w_lr(n, lr, ldn, rdn, tf), w_lr(n, lr, ln, rn, tf),
            mlr(n, ldn, rdn), mlr(n, ln, rn),
            w_lr(n, mlr, ldn, rdn, tf), w_lr(n, mlr, ln, rn, tf),
            mc_value(n)
        )

def debug_bigram():
    for a in bigram_freq.items():
        for b in a[1].items():
            print a[0], b[0], b[1]

## 文書解析関数

def analyze(corpus, corpus_file='*.txt', corpus_for_idf=None, corpus_for_idf_file='*.txt'):
    analyze_corpus(corpus, corpus_file)
    if corpus_for_idf:
        analyze_corpus_for_optimize_idf(corpus_for_idf, corpus_for_idf_file)

def analyze_corpus(corpus_dir, corpus_file='*.txt'):
    """平文コーパスから単語と単語バイグラムの出現頻度を計算する。"""
    if db is None:
        open_db()
    print 'Analyzing corpus in ' + corpus_dir
    for f in glob.glob(corpus_dir + '/' + corpus_file):
        with codecs.open(f, 'r', 'utf-8') as fi:
            print '  {} ...'.format(os.path.basename(f)) 
            for doc in fi:
                if not doc.startswith('#!'):
                    analyze_doc(doc)
    flush_freq(True)

def analyze_doc(text):
    """テキストに含まれる単語と単語バイグラムの出現頻度を更新する"""
    global numdoc
    global term_freq
    global bigram_freq
    global doc_freq

    left = ''
    term = ''
    feat = ''
    docterms = []
    
    encoded_text = text.encode('utf-8')
    node = mecab.parseToNode(encoded_text)
    
    while node:
        nfeat = node.feature.split(',')
        if is_under_concern_feat(nfeat):
            if left:
                bigram_freq[left][node.surface] += 1
            left = node.surface
            term = term + '_' + node.surface
            feat = feat + '_' + ','.join(nfeat[0:3]) 
        else:
            if is_under_concern_term(term, feat):
                term = term.strip('_')
                docterms.append(term)
                term_freq[term] += 1
            left = ''
            term= ''
            feat = ''
        node = node.next 
    
    for t in set(docterms):
        doc_freq[t] += 1
    
    numdoc += 1
    
    flush_freq()

def analyze_corpus_for_optimize_idf(corpus_dir, corpus_file='*.txt'):
    """
        分野特化したコーパスにおいて、IDF値を最適化するために
        参照コーパスを用いて文書頻度の一般化を行う
    """
    print 'Analyzing corpus in ' + corpus_dir
    for f in glob.glob(corpus_dir + '/' + corpus_file):
        with codecs.open(f, 'r', 'utf-8') as fi:
            print '  {} ...'.format(os.path.basename(f))
            for doc in fi:
                if not doc.startswith('#!'):
                    analyze_doc_for_optimize_idf(doc)
    flush_doc_freq(True)

def analyze_doc_for_optimize_idf(text):
    """語文書頻度の更新"""
    global numdoc
    global doc_freq

    term = ''
    feat = ''
    docterms = []
    
    encoded_text = text.encode('utf-8')
    node = mecab.parseToNode(encoded_text)
     
    while node:
        nfeat = node.feature.split(',')
        if is_under_concern_feat(nfeat):
            term = term + '_' + node.surface
            feat = feat + '_' + (','.join(nfeat[0:3]))
        else:
            if is_under_concern_term(term, feat):
                term = term.strip('_')
                docterms.append(term)
            term= ''
            feat = ''
        node = node.next
    
    for t in set(docterms):
        doc_freq[t] += 1
  
    numdoc += 1 

def is_under_concern_feat(feats):
    return ','.join(feats[0:4]) in accept_feature

def is_under_concern_term(term, feat):
    if not term:
        return False
    else:
        for r in except_feature_patterns:
            if re.match(r, feat):
                #print term, r
                return False
    return True

## DB更新関数

def flush_freq(force=False):
    flush_term_freq(force)
    flush_bigram_freq(force)
    flush_doc_freq(force)
        
def flush_term_freq(force=False):
    global term_freq
    global db
    if db_flush_threshould < len(term_freq) or force:
        for t in filter(lambda x:'_' in x, term_freq.keys()):
            db.execute(u'insert or ignore into term(term, freq) values (?, 0)', (t,))
            db.execute(u'update term set freq = freq + ? where term = ?', (term_freq[t], t))
        term_freq.clear()

def flush_bigram_freq(force=False):
    global bigram_freq
    global db
    if db_flush_threshould < len(bigram_freq) or force:
        for bi in [(e[0], v[0], v[1]) for e in bigram_freq.items() for v in e[1].items() if v[1]]:
            db.execute(u'insert or ignore into bigram (term1, term2, freq) values (?, ?, 0)', (bi[0], bi[1]))
            db.execute(u'update bigram set freq = freq + ? where term1 = ? and term2 = ?', (bi[2], bi[0], bi[1]))
        bigram_freq.clear()

def flush_doc_freq(force=False):
    global numdoc
    global doc_freq
    global db
    if db_flush_threshould < len(doc_freq) or force:
        db.execute(u'delete from stats') 
        db.execute(u'insert into stats(numdoc) values (?)', (numdoc,))
        for t in filter(lambda x:'_' in x, doc_freq.keys()):
            db.execute(u'insert or ignore into termdoc(term, freq) values (?, 0)', (t,))
            db.execute(u'update termdoc set freq = freq + ? where term = ?', (doc_freq[t], t))
        doc_freq.clear()

## DBM関数

def open_db():
    global db
    global db_file
    db_file = dic_dir + '/lex.db'
    db = sqlite3.connect(db_file, isolation_level=None)
    db.execute(term_ddl)
    db.execute(bigram_ddl)
    db.execute(docfreq_ddl)
    db.execute(stats_ddl)
    db.execute(idx_termfreq_ddl)
    db.execute(idx_bigramfreq_ddl)
    db.execute(idx_docfreq_ddl)
    db.text_factory = str

def close_db():
    global db
    if db is None:
        return
    try:
        db.close()
        db = None
    except:
        pass

def clean_db():
    global db
    if db is None:
        open_db()
    db.execute(u'delete from term')
    db.execute(u'delete from bigram')
    db.execute(u'delete from termdoc')
    db.execute(u'delete from stats')

def drop_db():
    close_db()
    if os.path.exists(db_file):
        os.remove(db_file)

def clean_dict():
    global numdoc
    global term_freq
    global bigram_freq
    global doc_freq
    numdoc = 0
    term_freq.clear()
    bigram_freq.clear()
    doc_freq.clear()

## 連接名詞スコア関数

def ldn(noun):
    """連接種類数(L)"""
    return sqlf(u'select count(term1) from bigram where term2 = ?', (noun,))

def ldn_in_dict(noun):
    return len([left for left in bigram_freq.keys() if bigram_freq[left][noun]])

def rdn(noun):
    """連接種類数(R)"""
    return sqlf(u'select count(term2) from bigram where term1 = ?', (noun,))

def rdn_in_dict(noun):
    return len([right for right in bigram_freq[noun].keys() if bigram_freq[noun][right]]) 

def ln(noun):
    """連接頻度(L)"""
    return sqlf(u'select ifnull(sum(freq), 0) from bigram where term2 = ?', (noun,))

def ln_in_dict(noun):
    return sum([bigram_freq[left][noun] for left in bigram_freq.keys()])

def rn(noun):
    """連接頻度(R)"""
    return sqlf(u'select ifnull(sum(freq), 0) from bigram where term1 = ?', (noun,))

def rn_in_dict(noun):
    return sum(bigram_freq[noun].values())

def ldn_ln(noun):
    return math.pow(ldn(noun) * ln(noun), 1 / 2.0)

def rdn_rn(noun):
    return math.pow(rdn(noun) * rn(noun), 1 / 2.0)

## 複合名詞スコア関数

def lr(term, fl, fr):
    """LR関数 連接する名詞スコアの相乗平均(対数平均)"""
    score = 0
    nouns = term.split("_")
    for noun in nouns:
        score += math.log10(fl(noun) + 1)
        score += math.log10(fr(noun) + 1)
    return math.pow(10, float(score) / (2.0 * float(len(nouns))))

def mlr(term, fl, fr):
    cterm = term.replace('_', '')
    node = mecab.parseToNode(cterm)
    node = node.next
    noun_len = 0
    score = 0
    non_noun = 0
    while node:
        noun = node.surface
        noun_feat = node.feature
        noun_len += 1
        if not noun_feat.startswith('名詞'):
            non_noun += 1
        elif is_except_lr(noun, noun_feat):
            score += math.log10(2)
        else:
            score += math.log10(fl(noun) + 1)
            score += math.log10(fr(noun) + 1)
        node = node.next
    return 0 if non_noun > 2 else math.pow(10, float(score) / (2.0 * float(noun_len)))

def is_except_lr(noun, feat):
    return ','.join(feat.split(',')[0:4]) in except_lr_features or noun in except_lr_surfaces

def w_lr(term, f, fl, fr, fw=lambda x:1):
    """重み付きLR関数"""
    return fw(term) * f(term, fl, fr)

def mc_value(term):
    """Modified C-Value"""
    length = float(term.count("_") + 1)
    ltf = long_tfreqs(term)
    t = float(sum(ltf))
    c = float(len(ltf))
    n = float(tfreq(term)) + t
    return length * n if c == 0 else length * (n - t / c)

## 重み関数

def tfreq(term):
    """出現頻度"""
    return math.sqrt(sqlf(u'select ifnull(sum(freq), 0) from term where term like ?', ('%' + term + '%',)))

def tf(term):
    """単独出現頻度"""
    return math.sqrt(sqlf(u'select ifnull(freq, 0) from term where term = ?', (term,)))

def idfreq(term):
    """逆文書頻度"""
    return math.log(dnum() / (dfreq(term) + 1.0)) + 1

def idf(term):
    """単独逆文書頻度"""
    return math.log(dnum() / (df(term) + 1.0)) + 1

def tfidf(term):
    "TF-IDF"
    return tf(term) * idf(term)

## その他、ターム、文書特徴関数

def dnum():
    return sqlf(u'select ifnull(numdoc,0) from stats', ())

def dfreq(term):
    """文書出現頻度"""
    return sqlf(u'select ifnull(sum(freq), 0) from termdoc where term like ?', ('%' + term + '%',))

def df(term):
    """文書単独出現頻度"""
    return sqlf(u'select ifnull(freq, 0) from termdoc where term = ?', (term,))

def long_tfreqs(term):
    freqs = []
    cur = db.cursor()
    cur.execute(u'select term, freq from term where term like ?', ('%' + term + '%',))
    for row in cur:
        if row[0] != term: 
            freqs.append(row[1])
    return freqs

## ユーティリティ関数

def cache(fid, term):
    return result[fid][term]

def sqlf(sql, args):
    cur = db.cursor()
    cur.execute(sql, args)
    ret = cur.fetchone()
    if ret is None:
        return 0
    else:
        return ret[0]

def set_user_dic(path):
    global mecab
    mecab = MeCab.Tagger('-Ochasen -u ' + path)

## スコア計算実行関数

def scoring_lex():
    print 'Populate TF-IDF'
    run_tfidf()
    print 'Populate Pattern LR ...'
    run_lr(f=mlr, fl=ldn, fr=rdn, rid='dnlr')
    print ' TF-IDF Boosting ...'
    run_cached_lr('tfidf_dnlr', 'dnlr', 'tfidf')

def run_lr(f=lr, fl=ln, fr=rn, fw=lambda x:1, rid='lr'):
    f_score = init_result(rid)
    cur = db.cursor()
    cur.execute(u'select term from term')
    for row in cur:
        if '_' in row[0]:
            f_score[row[0]] = w_lr(row[0], f, fl, fr, fw)

def run_mc_value():
    run_simple('mcv', mc_value)

def run_tf():
    run_simple('tf', tf)

def run_idf():
    run_simple('idf', idf)

def run_tfidf():
    run_simple('tfidf', tfidf)

def run_simple(fid, f):
    f_score = init_result(fid)
    cur = db.cursor()
    cur.execute(u'select term from term')
    for row in cur:
        if '_' in row[0]:
            f_score[row[0]] = f(row[0])

def run_cached_lr(fid, lr, fw):
    f_score = init_result(fid)
    lr_cache = result[lr]
    fw_cache = result[fw]
    for term in lr_cache.items():
        f_score[term[0]] = term[1] * fw_cache[term[0]]

def init_result(fid):
    global result
    if result.has_key(fid):
        result[fid].clear()
    else:
        result[fid] = {}
    return result[fid]

## 結果出力関数

def topn(n=sys.maxint):
    for term, score in sorted(result.items(), key=lambda x: x[1], reverse=True)[:n]:
        print term, score

def write_dic(topn):
    write('tfidf_dnlr', dic_dir + '/lex.csv', topn)

def write(fid, path, n=sys.maxint):
    with codecs.open(path, 'w', 'utf-8') as f:
        for term, score in sorted(result[fid].items(), key=lambda x: x[1], reverse=True)[:n]:
            lex = term.replace('_', '')
            node = mecab.parseToNode(lex)
            pos = []
            base = []
            read = []
            proun = []
            while node:
                feats = node.feature.split(',')
                pos.append(','.join(feats[0:3]))
                if 7 <= len(feats):
                    base.append(''.join(feats[6]))
                if 8 <= len(feats):
                    read.append(''.join(feats[7]))
                if 9 <= len(feats):
                    proun.append(''.join(feats[8]))
                node = node.next
            try:
                f.write(u'{}\t{},*,*,*,{},{},{}\n'.format(
                    lex,
                    predict_pos(pos),
                    ''.join(base[1:len(base)-1]),
                    ''.join(read[1:len(read)-1]),
                    ''.join(proun[1:len(read)-1]))) 
            except UnicodeDecodeError, detail:
                print 'UnicodeError at ' + term, detail
        f.write('EOS')

def predict_pos(pos):
    ## TODO
    return u'名詞,固有名詞,一般'

## main

def gen_dic(corpus_dir, output_dir, topn, corpus_file='*.txt', corpus_for_idf=None, corpus_for_idf_file='*.txt'):
    print 'Extracting lexes from {} in {}'.format(corpus_file, corpus_dir)
    if corpus_for_idf:
        print ' Using {} in {} for optimize IDF'.format(corpus_for_idf_file, corpus_for_idf)
    
    global dic_dir
    dic_dir = output_dir

    analyze(corpus_dir, corpus_file, corpus_for_idf, corpus_for_idf_file)

    scoring_lex()

    write_dic(topn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''
            lexgen: 未知語抽出器

            例: コーパスの平文テキストから抽出された上位100件の未知語からCSV辞書を生成する
                python script/lexgen.py -n 100 corpus/sample_doc/seg0/pln/ corpus/sample_doc/dict/
        ''')
        )
    parser.add_argument('text_dir', metavar='TEXT DIRECTORY',
                        nargs='?', help='平文テキストディレクトリ')
    parser.add_argument('output_dir', metavar='OUTPUT DIRECTORY',
                        nargs='?', help='CSV出力ディレクトリ')
    parser.add_argument('-n', '--topn', type=int, default=100,
                        help='抽出件数、スコア上位N件')
    args = parser.parse_args()

    if args.text_dir is None or args.output_dir is None:
        parser.print_usage()
        sys.exit(0)

    gen_dic(args.text_dir, args.output_dir, args.topn)
