#!/bin/sh

n=1
for f in `ls -tr *.jpg`; do
    name="`printf '%.2d' $n`"
    echo $name
    mv "$f" "$name".jpg
    n=`expr $n + 1`
done
