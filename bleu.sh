infile=$1
outfile=$2

cat ${infile} | awk -F '\t' '{print $2}' | sed -n '1,$p;n' > valid.truth
cat ${infile} | awk -F '\t' '{print $2}' | sed -n '1,$n;p' > valid.sample

./multi-bleu.perl valid.truth < valid.sample > nohup.bleu.tmp
sort -n nohup.bleu.tmp > ${outfile}

rm valid.truth valid.sample nohup.bleu.tmp

