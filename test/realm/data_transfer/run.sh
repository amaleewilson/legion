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
	)

output_file=perfdata_$1$(date +"%Y%m%d%H%M").csv

echo "method,time,fieldIDcount,elemcount,bytes,bytesPerNano" > $output_file

for n in ${nelems[*]};
do

        for i in {1..10};
        do
            echo "$(./$1 $n)" >> $output_file;
        done

done



