#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

typedef double T;

// Вариант А
void transpose_rows(const T *A, T *B, size_t N) {
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            B[j * N + i] = A[i * N + j];
}

void transpose_cols(const T *A, T *B, size_t N) {
    for (size_t j = 0; j < N; j++)
        for (size_t i = 0; i < N; i++)
            B[j * N + i] = A[i * N + j];
}

void transpose_blocked(const T *A, T *B, size_t N, size_t b) {
    for (size_t ii = 0; ii < N; ii += b) {
        size_t imax = (ii + b < N) ? ii + b : N;
        for (size_t jj = 0; jj < N; jj += b) {
            size_t jmax = (jj + b < N) ? jj + b : N;
            for (size_t i = ii; i < imax; i++)
                for (size_t j = jj; j < jmax; j++)
                    B[j * N + i] = A[i * N + j];
        }
    }
}

void transpose_rows_omp(const T *A, T *B, size_t N) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)N; i++)
        for (size_t j = 0; j < N; j++)
            B[j * N + (size_t)i] = A[(size_t)i * N + j];
}

void transpose_rows_omp_chunk(const T *A, T *B, size_t N, int chunk) {
    #pragma omp parallel for schedule(static, chunk)
    for (long long i = 0; i < (long long)N; i++)
        for (size_t j = 0; j < N; j++)
            B[j * N + (size_t)i] = A[(size_t)i * N + j];
}

void transpose_blocked_omp(const T *A, T *B, size_t N, size_t b) {
    long long nblocks = (long long)((N + b - 1) / b);
    #pragma omp parallel for schedule(static)
    for (long long bi = 0; bi < nblocks; bi++) {
        size_t ii = (size_t)bi * b;
        size_t imax = (ii + b < N) ? ii + b : N;
        for (size_t jj = 0; jj < N; jj += b) {
            size_t jmax = (jj + b < N) ? jj + b : N;
            for (size_t i = ii; i < imax; i++)
                for (size_t j = jj; j < jmax; j++)
                    B[j * N + i] = A[i * N + j];
        }
    }
}

void run(const char *v, const T *A, T *B, size_t N, size_t b) {
    if (strcmp(v, "rows") == 0) transpose_rows(A, B, N);
    else if (strcmp(v, "cols") == 0) transpose_cols(A, B, N);
    else if (strcmp(v, "blocked") == 0) transpose_blocked(A, B, N, b);
    else if (strcmp(v, "rows_omp") == 0) transpose_rows_omp(A, B, N);
    else if (strcmp(v, "rows_omp_static1") == 0) transpose_rows_omp_chunk(A, B, N, 1);
    else if (strcmp(v, "rows_omp_static8") == 0) transpose_rows_omp_chunk(A, B, N, 8);
    else if (strcmp(v, "blocked_omp") == 0) transpose_blocked_omp(A, B, N, b);
    else { fprintf(stderr, "unknown version: %s\n", v); exit(1); }
}

double median5(double *arr) {
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 5; j++)
            if (arr[j] < arr[i]) {
                double t = arr[i]; arr[i] = arr[j]; arr[j] = t;
            }
    return arr[2];
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s N b threads version\n", argv[0]);
        return 1;
    }
    size_t N = (size_t)atoll(argv[1]);
    size_t b = (size_t)atoll(argv[2]);
    int threads = atoi(argv[3]);
    const char *version = argv[4];

    omp_set_num_threads(threads);

    T *A = (T*)malloc(N * N * sizeof(T));
    T *B = (T*)malloc(N * N * sizeof(T));
    T *Bref = (T*)malloc(N * N * sizeof(T));
    if (!A || !B || !Bref) { fprintf(stderr, "malloc failed\n"); return 1; }

    for (size_t k = 0; k < N * N; k++) A[k] = (T)k;
    transpose_rows(A, Bref, N);

    run(version, A, B, N, b);

    double times[5];
    for (int r = 0; r < 5; r++) {
        int k = 0;
        double t = omp_get_wtime();
        do {
            run(version, A, B, N, b);
            k++;
        } while (omp_get_wtime() - t < 0.1);
        times[r] = (omp_get_wtime() - t) / k;
    }
    double med = median5(times);

    size_t bad = 0;
    for (size_t k = 0; k < N * N; k++)
        if (B[k] != Bref[k]) bad++;

    double bw = 2.0 * (double)N * (double)N * sizeof(T) / med / 1e9;
    int used = (strstr(version, "omp") != NULL) ? threads : 1;

    printf("N=%zu b=%zu threads=%d version=%s time=%.6f s BW=%.2f GB/s check=%s\n",
           N, b, used, version, med, bw, bad == 0 ? "OK" : "FAIL");

    free(A); free(B); free(Bref);
    return 0;
}