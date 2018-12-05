IFS=""

nelems=(
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
    "no_trans"
  )

# $1 is kernel_transpose

for m in ${methods[*]};
do
for n in ${nelems[*]};
do
    ./$1 -ne $n -method $m

done
done
