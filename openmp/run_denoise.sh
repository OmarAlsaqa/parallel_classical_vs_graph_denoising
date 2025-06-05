#!/bin/bash

# Usage: ./run_denoise.sh <input_image> <noising_rate> <alpha> <iterations> <resize> <processes>
# Example: ./run_denoise.sh input.png 0.01 0.1 10 yes 4

input_image=$1  # Accept input image name as an argument
noising_rate=$2  # Accept noising rate as an argument
alpha=$3  # Accept alpha as an argument
iterations=$4  # Accept number of iterations as an argument
resize=$5            # Resize option: "yes" or "no"
num_processes=$6     # Number of MPI processes

# Set the number of OpenMP threads
export OMP_NUM_THREADS=$num_processes

output_prefix=${input_image%.*}  # Base name without extension

# Conditional resize
if [ "$resize" == "yes" ]; then
    convert "$input_image" -resize 4096x4096 "${output_prefix}.ppm"
else
    convert "$input_image" "${output_prefix}.ppm"
fi

gcc -O3 -std=c99 -fopenmp -o add_noise add_noise.c -lm

./add_noise "${output_prefix}.ppm" noisy_output.ppm "$noising_rate"

convert noisy_output.ppm noisy_output.png

# Run graph-based denoising
gcc -O3 -std=c99 -fopenmp -o graph_denoise_rgb graph_denoise_rgb.c -lm

runs=10  # Number of runs for averaging
total_sum=0

for i in $(seq 1 $runs); do
    output=$(./graph_denoise_rgb noisy_output.ppm graph_denoised_output.ppm "$alpha" "$iterations")
    # Extract the total value
    total_value=$(echo "$output" | grep -oP 'Total.*?in\s+\K[0-9.]+')
    # Add to the sum
    total_sum=$(echo "$total_sum + $total_value" | bc)
done

# Calculate average
average=$(echo "scale=4; $total_sum / $runs" | bc)
echo "Graph average over $runs runs: $average"

convert graph_denoised_output.ppm graph_denoised_output.png

# Run median-based denoising
gcc -O3 -std=c99 -fopenmp -o median_denoise_rgb median_denoise_rgb.c -lm

total_sum=0

for i in $(seq 1 $runs); do
    output=$(./median_denoise_rgb noisy_output.ppm median_denoised_output.ppm)
    # Extract the total value
    total_value=$(echo "$output" | grep -oP 'Total.*?in\s+\K[0-9.]+')
    # Add to the sum
    total_sum=$(echo "$total_sum + $total_value" | bc)
done

# Calculate average
average=$(echo "scale=4; $total_sum / $runs" | bc)
echo "Median average over $runs runs: $average"

convert median_denoised_output.ppm median_denoised_output.png

# Clean up intermediate files
rm -f graph_denoise_rgb median_denoise_rgb add_noise
rm -f "${output_prefix}.ppm" noisy_output.ppm graph_denoised_output.ppm median_denoised_output.ppm