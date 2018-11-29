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

output_file=data/perfdata_$1$(date +"%Y%m%d%H%M").csv

echo "method,time,fieldIDcount,elemcount,bytes,bytesPerNano,machine" > $output_file

for n in ${nelems[*]};
do

        for i in {1..10};
        do
            echo "$(./$1 -ne $n),$2" >> $output_file;
        done

done



