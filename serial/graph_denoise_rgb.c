#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

// Enhanced edge-aware graph diffusion
void graph_diffusion_rgb(PPMImage *input, PPMImage *output, float alpha, int iterations) {
    unsigned char *temp = (unsigned char*)malloc(input->width * input->height * 3);
    memcpy(temp, input->data, input->width * input->height * 3);

    int width = input->width, height = input->height;
    float sigma = 20.0f, threshold = 20.0f;

    for (int iter = 0; iter < iterations; iter++) {
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                for (int c = 0; c < 3; c++) {
                    int idx = (y * width + x) * 3 + c;
                    int center = temp[idx];

                    int neighbors[4] = {
                        temp[((y - 1) * width + x) * 3 + c],
                        temp[((y + 1) * width + x) * 3 + c],
                        temp[(y * width + (x - 1)) * 3 + c],
                        temp[(y * width + (x + 1)) * 3 + c]
                    };

                    float weight_sum = 0.0f, weighted_value = 0.0f;
                    for (int i = 0; i < 4; i++) {
                        float diff = neighbors[i] - center;
                        float weight = expf(-(diff * diff) / (2 * sigma * sigma));
                        weight_sum += weight;
                        weighted_value += weight * neighbors[i];
                    }

                    float smooth_value = weighted_value / weight_sum;
                    float diff = fabsf(smooth_value - center);

                    float result = (diff > threshold) ? smooth_value : center + alpha * (smooth_value - center);
                    output->data[idx] = (unsigned char)(fminf(fmaxf(result, 0), 255));
                }
            }
        }
        unsigned char *swap = temp;
        temp = output->data;
        output->data = swap;
    }

    if (iterations % 2 == 1) {
        memcpy(output->data, temp, width * height * 3);
    }

    free(temp);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <input.ppm> <output.ppm> <alpha> <iterations>\n", argv[0]);
        return 1;
    }

    float alpha = atof(argv[3]);
    int iterations = atoi(argv[4]);
    if (iterations <= 0) {
        fprintf(stderr, "Iterations must be a positive integer.\n");
        return 1;
    }

    clock_t total_start_time = clock();

    PPMImage *input = read_ppm(argv[1]);
    if (!input) return 1;

    PPMImage *output = (PPMImage*)malloc(sizeof(PPMImage));
    output->width = input->width;
    output->height = input->height;
    output->data = (unsigned char*)malloc(input->width * input->height * 3);

    clock_t start_time = clock();
    graph_diffusion_rgb(input, output, alpha, iterations);
    printf("Graph filtering completed %.4f seconds.\n", 
           (double)(clock() - start_time) / CLOCKS_PER_SEC);

    write_ppm(argv[2], output);

    printf("Total (graph) process completed in %.4f seconds.\n", 
           (double)(clock() - total_start_time) / CLOCKS_PER_SEC);

    free(input->data);
    free(input);
    free(output->data);
    free(output);

    return 0;
}
