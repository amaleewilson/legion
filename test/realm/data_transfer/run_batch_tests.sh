
IFS=""

nelems=(
    "33554432"
  )
#  "1024"
#    "2048"
#    "4096"
#    "8192"
#    "16384"
#    "32768"
#    "65536"
#    "131072"
#    "262144"
#    "524288"
#    "1048576"
#    "2097152"
#    "4194304"
#    "8388608"
#    "16777216"

batch_methods_multi=(
    "bp_copy"
)
    #"trans_multi_batch"
    #"trans2_multi_batch"


batch_methods=(
    "trans1_batch"
    "trans2_batch"
  )

methods=(
  "trans1"
    "trans2"
    "trans_multi8"
    "trans_multi4"
    "share_trans_multi8"
    "share_trans_multi4"
    "share_trans"
    "memcpy_no_trans"
  )

  block_sizes=(
    "1"
    "2"
    "4"
    "8"
    "16"
    "32"
    "64"
    "128"
    "256"
    "512"
  )

  copy_count=(
    "128"
    "256"
  )


  # this one is so bad, no reason to test it. 
  #"memcpy_trans1"

# $1 is kernel_transpose
# $2 is machine

output_file=data/data_$(date +"%Y%m%d%H%M").csv

echo "method,time,fieldIDcount,elemcount,bytes,bytesPerNano,block_count,block_size,copy_count,machine" > $output_file

: '
for m in ${batch_methods[*]};
do
for n in ${nelems[*]};
do
    for bsize in ${block_sizes[*]};
    do
        for i in {1..10};
        do
            
            echo "$(./$1 -ne $n -method $m -block_count $bsize),$2" >> $output_file;
        
        done
    done
done
done
'

for m in ${batch_methods_multi[*]};
do
for n in ${nelems[*]};
do
    for c_sz in ${copy_count[*]};
    do
    for bsize in ${block_sizes[*]};
    do
        for i in {1..10};
        do
            
            echo "$(./$1 -ne $n -method $m -block_count $bsize -copy_count $c_sz),$2" >> $output_file;
        
        done
    done
  done
done
done



: '
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
'



