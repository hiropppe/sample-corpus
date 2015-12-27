#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import codecs
import argparse
import textwrap
from os.path import join

def clf(f, tag, out):
    doc_id = None
    bib_id = None
    tag_val = None 
    sents = []
    is_in_sent = False
    with codecs.open(f, 'r', 'utf-8') as f:
        for l in f:
            # #! DOC 0
            m = re.match('#!\sDOC\s([0-9]+)', l) 
            if not m is None:
                write_doc(bib_id, tag, tag_val, sents, out)

                doc_id = int(m.group(1))
                tag_val = None
                sents = []
                continue
            
            # #! ATTR "Bib_ID" "100000010133" ""
            bib_r = r'#! ATTR "Bib_ID" "(.+?)"'
            m = re.match(bib_r, l)
            if not m is None:
                bib_id = m.group(1)
                continue

            # Target Tag
            # #! ATTR "Harmful" "1" ""
            tag_r = r'#! ATTR "{}" "(.+?)"'.format(tag)
            m = re.match(tag_r, l)
            if not m is None:
                tag_val = m.group(1)
                continue
            
            # * 0 -1D 0/0 0
            r_bos = r'\* \d \-\dD \d\/\d \d'
            m = re.match(r_bos, l)
            if not m is None:
                is_in_sent = True
                continue
            
            # EOS
            r_eos = r'^EOS'
            m = re.match(r_eos, l)
            if not m is None:
                is_in_sent = False
                continue
            
            # 形態素 
            if is_in_sent:
                sents.append(re.split(r'\t', l)[0])

    write_doc(bib_id, tag, tag_val, sents, out)

def write_doc(bib_id, tag, tag_val, sents, out):
    if bib_id is None: return
    if tag_val is None: return
    if len(sents) == 0: return
    
    print tag, tag_val, bib_id
    
    dir = join(out, tag, tag_val)
    path = join(dir, bib_id + '.txt')
    
    if not os.path.exists(dir):
        os.makedirs(dir)

    with codecs.open(path, 'w', 'utf-8') as f:
        f.write(''.join(sents))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''
        Usage: python script/chaki_clf_corpus_gen.py path/to/chaki/export/all.cabocha Harmful .
        ''')
        )
    parser.add_argument('chaki_export_file')
    parser.add_argument('clf_doc_tag')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    clf(args.chaki_export_file, args.clf_doc_tag, args.output_dir)

