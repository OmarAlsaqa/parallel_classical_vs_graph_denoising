#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct
{
    int width;
    int height;
    unsigned char *data; // RGB data stored as [R, G, B, R, G, B, ...]
} PPMImage;

// Read PPM (P6 format)
PPMImage *read_ppm(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        perror("Error opening file");
        return NULL;
    }

    PPMImage *img = (PPMImage *)malloc(sizeof(PPMImage));
    char version[3];
    if (fscanf(fp, "%2s", version) != 1)
    {
        fprintf(stderr, "Error reading PPM version\n");
        fclose(fp);
        free(img);
        return NULL;
    }
    if (version[1] != '6')
    {
        fprintf(stderr, "Only P6 supported\n");
        fclose(fp);
        free(img);
        return NULL;
    }

    if (fscanf(fp, "%d %d %*d", &img->width, &img->height) != 2)
    {
        fprintf(stderr, "Error reading image dimensions\n");
        fclose(fp);
        free(img);
        return NULL;
    }
    fgetc(fp); // Skip newline

    img->data = (unsigned char *)malloc(img->width * img->height * 3);
    if (fread(img->data, 1, img->width * img->height * 3, fp) != img->width * img->height * 3)
    {
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
void write_ppm(const char *filename, PPMImage *img)
{
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P6\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, 1, img->width * img->height * 3, fp);
    fclose(fp);
}

// Median filter for RGB image (3x3 kernel) using MPI
void median_filter_rgb_parallel(PPMImage *input, PPMImage *output, int rank, int size)
{

    int width = input->width, height = input->height;
    int rows_per_process = height / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? height : start_row + rows_per_process;

    unsigned char *temp = (unsigned char *)malloc(width * (end_row - start_row) * 3);

    for (int y = start_row; y < end_row; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            for (int c = 0; c < 3; c++)
            { // Process each channel (R, G, B)
                unsigned char window[9];
                int idx = 0;

                // Collect 3x3 neighborhood for the current channel
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int neighbor_idx = ((y + dy) * width + (x + dx)) * 3 + c;
                        window[idx++] = input->data[neighbor_idx];
                    }
                }

                // Bubble sort (for simplicity)
                for (int i = 0; i < 9; i++)
                {
                    for (int j = i + 1; j < 9; j++)
                    {
                        if (window[i] > window[j])
                        {
                            unsigned char tmp = window[i];
                            window[i] = window[j];
                            window[j] = tmp;
                        }
                    }
                }

                // Set the median value for the current channel
                int output_idx = ((y - start_row) * width + x) * 3 + c;
                temp[output_idx] = window[4]; // Median
            }
        }
    }

    MPI_Gather(temp, width * (end_row - start_row) * 3, MPI_UNSIGNED_CHAR,
               output->data, width * rows_per_process * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    free(temp);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    double total_start_time = MPI_Wtime(); // Start timing for the entire program

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3)
    {
        if (rank == 0)
            printf("Usage: %s <input.ppm> <output.ppm>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    PPMImage *input = NULL, *output = NULL;
    if (rank == 0)
    {
        input = read_ppm(argv[1]);
        if (!input)
        {
            // Signal other processes to terminate gracefully
            int error_flag = 1;
            MPI_Bcast(&error_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Finalize();
            return 1;
        }
        // Signal successful read
        int error_flag = 0;
        MPI_Bcast(&error_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else
    {
        // Wait for signal from rank 0
        int error_flag;
        MPI_Bcast(&error_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (error_flag)
        {
            MPI_Finalize();
            return 1;
        }
        input = (PPMImage *)malloc(sizeof(PPMImage));
    }

    int width, height;
    if (rank == 0)
    {
        width = input->width;
        height = input->height;
    }
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Set output dimensions and allocate data buffer on all ranks
    output = (PPMImage *)malloc(sizeof(PPMImage));
    output->width = width;
    output->height = height;
    output->data = (unsigned char *)malloc(width * height * 3);

    if (rank != 0)
    {
        input = (PPMImage *)malloc(sizeof(PPMImage));
        input->width = width;
        input->height = height;
        input->data = (unsigned char *)malloc(width * height * 3);
    }
    MPI_Bcast(input->data, width * height * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    double compute_start_time = MPI_Wtime();
    median_filter_rgb_parallel(input, output, rank, size);
    double compute_end_time = MPI_Wtime();

    if (rank == 0)
    {
        write_ppm(argv[2], output);
    }

    // Free resources on all ranks
    free(input->data);
    free(input);
    free(output->data);
    free(output);

    double total_end_time = MPI_Wtime(); // End timing for the entire program
    if (rank == 0)
    {
        printf("Computation time (median_filter_rgb_parallel) in %.4f seconds.\n", compute_end_time - compute_start_time);
        printf("Total (Median) execution time in %f seconds\n", total_end_time - total_start_time);
    }

    MPI_Finalize();
    return 0;
}