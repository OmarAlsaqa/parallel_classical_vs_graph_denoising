#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

// Median filter for RGB image (3x3 kernel)
void median_filter_rgb(PPMImage *input, PPMImage *output) {
    for (int y = 1; y < input->height - 1; y++) {
        for (int x = 1; x < input->width - 1; x++) {
            for (int c = 0; c < 3; c++) { // Process each channel (R, G, B)
                unsigned char window[9];
                int idx = 0;

                // Collect 3x3 neighborhood for the current channel
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int neighbor_idx = ((y + dy) * input->width + (x + dx)) * 3 + c;
                        window[idx++] = input->data[neighbor_idx];
                    }
                }

                // Bubble sort (for simplicity)
                for (int i = 0; i < 9; i++) {
                    for (int j = i + 1; j < 9; j++) {
                        if (window[i] > window[j]) {
                            unsigned char tmp = window[i];
                            window[i] = window[j];
                            window[j] = tmp;
                        }
                    }
                }

                // Set the median value for the current channel
                int output_idx = (y * input->width + x) * 3 + c;
                output->data[output_idx] = window[4]; // Median
            }
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

    // Start timing for median filtering
    clock_t start_time = clock();

    // Perform median filtering
    median_filter_rgb(input, output);

    // End timing for median filtering
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Median filtering completed in %.4f seconds.\n", elapsed_time);

    // Write output image
    write_ppm(output_file, output);

    // End timing for the entire process
    clock_t total_end_time = clock();
    double total_elapsed_time = (double)(total_end_time - total_start_time) / CLOCKS_PER_SEC;
    printf("Total (median) process completed in %.4f seconds.\n", total_elapsed_time);

    // Free memory
    free(input->data);
    free(input);
    free(output->data);
    free(output);

    return 0;
}