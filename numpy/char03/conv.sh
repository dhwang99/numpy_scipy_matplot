awk -F"," '{if (FNR > 1 && $7 != 0 && $5 != 0 && ($5 / $7) > 4) {k3=$3/5; k4=$4/5; k2=$2/5;k5=$5/5; print $1","k2","k3","k4","k5","$6","$7} else print $0;}' data.csv > data2.csv
