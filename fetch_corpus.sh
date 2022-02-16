#!/usr/bin/env bash
# English(en), Chinese(zh), Hindi(hi), Spanish(es), French(fr), Arabic(ar), Bengali(bn), Russian(ru), Portuguese(pt), Indonesian(id)
DIR=`pwd`

echo $DIR

rm -rf wiki
mkdir wiki
cd wiki
for LANG in en-zh en-hi en-es en-fr ar-en bn-en en-ru en-pt en-id
do
wget https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.$LANG.tsv.gz
gunzip WikiMatrix.$LANG.tsv.gz
done

cd $DIR
