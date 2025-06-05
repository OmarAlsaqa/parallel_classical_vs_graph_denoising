#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


typedef struct {
    int width;
    int height;
    unsigned char *data; // RGB data stored as [R, G, B, R, G, B, ...]
} PPMImage;

// Read PPM (P6 format)
PPMImage* read_ppm(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) { perror("Error opening file"); return NULL; }

    PPMImage *img = (PPMImage*)malloc(sizeof(PPMImage));
    char version[3];
    if (fscanf(fp, "%2s", version) != 1) {
        fprintf(stderr, "Error reading PPM version\n");
        fclose(fp);
        free(img);
        return NULL;
    }
    if (version[1] != '6') {
        fprintf(stderr, "Only P6 supported\n");
        fclose(fp);
        free(img);
        return NULL;
    }

    if (fscanf(fp, "%d %d %*d", &img->width, &img->height) != 2) {
        fprintf(stderr, "Error reading image dimensions\n");
        fclose(fp);
        free(img);
        return NULL;
    }
    fgetc(fp); // Skip newline

    img->data = (unsigned char*)malloc(img->width * img->height * 3);
    if (fread(img->data, 1, img->width * img->height * 3, fp) != img->width * img->height * 3) {
        fprintf(stderr, "Error reading image data\n");
        fclose(fp);
        free(img->data);
        free(img);
        return NULL;
    }
    fclose(fp);
    return img;
}

// Write PPM (P6 format)
void write_ppm(const char *filename, PPMImage *img) {
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P6\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, 1, img->width * img->height * 3, fp);
    fclose(fp);
}
__device__ void bubble_sort(unsigned char *window, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            if (window[i] > window[j]) {
                unsigned char temp = window[i];
                window[i] = window[j];
                window[j] = temp;
            }
        }
    }
}

__global__ void median_filter_kernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        for (int c = 0; c < 3; c++) {
            unsigned char window[9];
            int idx = 0;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int neighbor_idx = ((y + dy) * width + (x + dx)) * 3 + c;
                    window[idx++] = input[neighbor_idx];
                }
            }

            bubble_sort(window, 9);
            int output_idx = (y * width + x) * 3 + c;
            output[output_idx] = window[4]; // Median
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input.ppm> <output.ppm>\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];

    // Start timing for the entire process
    clock_t total_start_time = clock();

    // Read input image
    PPMImage *input = read_ppm(input_file);
    if (!input) return 1;
    
    // Allocate output image
    PPMImage *output = (PPMImage*)malloc(sizeof(PPMImage));
    output->width = input->width;
    output->height = input->height;
    output->data = (unsigned char*)malloc(input->width * input->height * 3);

    
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, input->width * input->height * 3);
    cudaMalloc(&d_output, input->width * input->height * 3);
    cudaMemcpy(d_input, input->data, input->width * input->height * 3, cudaMemcpyHostToDevice);
    
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((input->width + 15) / 16, (input->height + 15) / 16);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);
    median_filter_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, input->width, input->height);
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Median Kernel execution time: %.6f seconds.\n", milliseconds/1000);

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(output->data, d_output, input->width * input->height * 3, cudaMemcpyDeviceToHost);

    // Write output image
    write_ppm(output_file, output);

    // End timing for the entire process
    clock_t total_end_time = clock();
    double total_elapsed_time = (double)(total_end_time - total_start_time) / CLOCKS_PER_SEC;
    printf("Total (Median) process completed in %.4f seconds.\n", total_elapsed_time);

    // Free memory
    free(input->data);
    free(input);
    free(output->data);
    free(output);
    
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
