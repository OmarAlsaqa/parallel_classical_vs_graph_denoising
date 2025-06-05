#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h> // Include OpenMP header

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

// Enhanced edge-aware graph diffusion (MPI + OpenMP version)
void graph_diffusion_rgb_parallel(PPMImage *input, PPMImage *output, float alpha, int iterations, int rank, int size)
{
    int width = input->width, height = input->height;
    size_t image_size = width * height * 3 * sizeof(unsigned char);
    unsigned char *curr = malloc(image_size);
    unsigned char *next = malloc(image_size);
    memcpy(curr, input->data, image_size);

    // Determine global block decomposition: each process works on rows [local_start, local_end)
    int rows_per_proc = height / size;
    int extra = height % size;
    int local_start, local_rows;
    if (rank < extra)
    {
        local_rows = rows_per_proc + 1;
        local_start = rank * local_rows;
    }
    else
    {
        local_rows = rows_per_proc;
        local_start = rank * rows_per_proc + extra;
    }
    int local_end = local_start + local_rows; // global row indices

    // Compute effective update region (skip global boundaries)
    int local_eff_start = (local_start < 1) ? 1 : local_start;
    int local_eff_end = (local_end > height - 1) ? height - 1 : local_end;
    int local_count = (local_eff_end - local_eff_start) * width * 3;

    // Build global recvcounts and displacements based on effective regions
    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        int proc_rows = (i < extra) ? (rows_per_proc + 1) : rows_per_proc;
        int proc_start = (i < extra) ? i * (rows_per_proc + 1) : i * rows_per_proc + extra;
        int proc_end = proc_start + proc_rows;
        int eff_start = (proc_start < 1) ? 1 : proc_start;
        int eff_end = (proc_end > height - 1) ? height - 1 : proc_end;
        recvcounts[i] = (eff_end - eff_start) * width * 3;
        displs[i] = eff_start * width * 3;
    }

    for (int iter = 0; iter < iterations; iter++)
    {
        memcpy(next, curr, image_size);
        // Update only interior rows within the local block
        #pragma omp parallel for collapse(2)
        for (int y = local_eff_start; y < local_eff_end; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    int idx = (y * width + x) * 3 + c;
                    int center = curr[idx];
                    int neighbors[4] = {
                        curr[((y - 1) * width + x) * 3 + c],
                        curr[((y + 1) * width + x) * 3 + c],
                        curr[(y * width + (x - 1)) * 3 + c],
                        curr[(y * width + (x + 1)) * 3 + c]};
                    float sigma = 20.0f, threshold = 20.0f;
                    float weight_sum = 0.0f, weighted_value = 0.0f;
                    for (int i = 0; i < 4; i++)
                    {
                        float diff = neighbors[i] - center;
                        float weight = expf(-(diff * diff) / (2 * sigma * sigma));
                        weight_sum += weight;
                        weighted_value += weight * neighbors[i];
                    }
                    float smooth_value = weighted_value / weight_sum;
                    float diff_val = fabsf(smooth_value - center);
                    float result = (diff_val > threshold) ? smooth_value : center + alpha * (smooth_value - center);
                    next[idx] = (unsigned char)(fminf(fmaxf(result, 0), 255));
                }
            }
        }
        // Gather only the effective interior region from every process into the full image buffer
        MPI_Allgatherv(next + (local_eff_start * width * 3),
                       local_count, MPI_UNSIGNED_CHAR,
                       curr, recvcounts, displs, MPI_UNSIGNED_CHAR,
                       MPI_COMM_WORLD);
    }
    memcpy(output->data, curr, image_size);
    free(curr);
    free(next);
    free(recvcounts);
    free(displs);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    double total_start_time = MPI_Wtime(); // Start timing for the entire program

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set number of threads for OpenMP (optional, often defaults to max available)
    int omp_threads = omp_get_max_threads();
    omp_set_num_threads(omp_threads); // Example: Use all available threads per process

    if (argc != 5)
    {
        if (rank == 0)
            printf("Usage: %s <input.ppm> <output.ppm> <alpha> <iterations>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    float alpha = atof(argv[3]);
    int iterations = atoi(argv[4]);
    if (iterations <= 0)
    {
        if (rank == 0)
            fprintf(stderr, "Iterations must be a positive integer.\n");
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
    graph_diffusion_rgb_parallel(input, output, alpha, iterations, rank, size);
    double compute_end_time = MPI_Wtime();

    if (rank == 0)
    {
        write_ppm(argv[2], output);
    }

    free(input->data);
    free(input);
    free(output->data);
    free(output);

    double total_end_time = MPI_Wtime(); // End timing for the entire program
    if (rank == 0)
    {
        printf("Computation time (graph_filter_rgb_parallel) in %.4f seconds.\n", compute_end_time - compute_start_time);
        printf("Total (Graph) execution time in %f seconds.\n", total_end_time - total_start_time);
    }

    MPI_Finalize();
    return 0;
}
