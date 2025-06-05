#include <stdio.h>
#include <stdlib.h>
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
        free(img);
        fclose(fp);
        return NULL;
    }
    if (version[1] != '6') { 
        fprintf(stderr, "Only P6 supported\n"); 
        free(img);
        fclose(fp);
        return NULL; 
    }

    if (fscanf(fp, "%d %d", &img->width, &img->height) != 2) {
        fprintf(stderr, "Error reading image dimensions\n");
        free(img);
        fclose(fp);
        return NULL;
    }
    int max_val;
    if (fscanf(fp, "%d", &max_val) != 1) {
        fprintf(stderr, "Error reading max value\n");
        free(img);
        fclose(fp);
        return NULL;
    }
    fgetc(fp); // Skip newline

    img->data = (unsigned char*)malloc(img->width * img->height * 3);
    if (fread(img->data, 1, img->width * img->height * 3, fp) != img->width * img->height * 3) {
        fprintf(stderr, "Error reading image data\n");
        free(img->data);
        free(img);
        fclose(fp);
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

// Add salt-and-pepper noise to RGB image
void add_salt_and_pepper_noise(PPMImage *img, float noise_prob) {
    srand(time(NULL)); // Seed random number generator
    int total_pixels = img->width * img->height;

    for (int i = 0; i < total_pixels; i++) {
        float rand_val = (float)rand() / RAND_MAX;

        if (rand_val < noise_prob / 2) {
            // Add salt noise (white pixel)
            img->data[i * 3 + 0] = 255;     // Red
            img->data[i * 3 + 1] = 255;     // Green
            img->data[i * 3 + 2] = 255;     // Blue
        } else if (rand_val < noise_prob) {
            // Add pepper noise (black pixel)
            img->data[i * 3 + 0] = 0;       // Red
            img->data[i * 3 + 1] = 0;       // Green
            img->data[i * 3 + 2] = 0;       // Blue
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <input.ppm> <output.ppm> <noise_probability>\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    float noise_prob = atof(argv[3]);

    if (noise_prob < 0 || noise_prob > 1) {
        fprintf(stderr, "Noise probability must be between 0 and 1.\n");
        return 1;
    }

    // Read input image
    PPMImage *img = read_ppm(input_file);
    if (!img) return 1;

    // Add salt-and-pepper noise
    add_salt_and_pepper_noise(img, noise_prob);

    // Write output image
    write_ppm(output_file, img);

    // Free memory
    free(img->data);
    free(img);
    printf("Salt-and-pepper noise added successfully.\n");

    return 0;
}