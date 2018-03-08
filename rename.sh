#!/bin/sh

IFS='\n'
n=1
for f in `ls -tr *.jpg`; do
    mv "$f" "$n".jpg
    n=`expr $n + 1`
done
