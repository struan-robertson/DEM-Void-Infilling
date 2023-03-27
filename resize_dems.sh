#!/usr/bin/env sh

# Input and output directories
input_dir="../Data"
output_dir="../Data_ds"

output_res=10

for input_file in ${input_dir}/*tif; do

    output_file="${output_dir}/$(basename ${input_file})"

    gdalwarp -tr ${output_res} ${output_res} -r cubic -overwrite ${input_file} ${output_file}

done
