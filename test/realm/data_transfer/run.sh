
IFS=""

nelems=(
  "1024"
    "2048"
    "4096"
    "8192"
    "16384"
    "32768"
    "65536"
    "131072"
    "262144"
    "524288"
    "1048576"
    "2097152"
    "4194304"
    "8388608"
    "16777216"
    "33554432"
  )

methods=(
  "trans1"
    "trans2"
    "trans_multi8"
    "trans_multi4"
    "share_trans_multi8"
    "share_trans_multi4"
    "share_trans"
    "memcpy_trans1"
    "memcpy_no_trans"
  )

# $1 is kernel_transpose
# $2 is machine

output_file=data/data_$(date +"%Y%m%d%H%M").csv

echo "method,time,fieldIDcount,elemcount,bytes,bytesPerNano,machine" > $output_file

for m in ${methods[*]};
do
for n in ${nelems[*]};
do
        for i in {1..10};
        do
            echo "$(./$1 -ne $n -method $m),$2" >> $output_file;
        done
done
done


