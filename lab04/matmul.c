#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

void mm_ijk_seq(const double *A, const double *B, double *C, int N)
{
    int i, j, k;
    double sum;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            sum = 0.0;
            for (k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

void mm_ijk_omp(const double *A, const double *B, double *C, int N)
{
    int i, j, k;
    double sum;
    #pragma omp parallel for private(j, k, sum)
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            sum = 0.0;
            for (k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

void mm_ikj_seq(const double *A, const double *B, double *C, int N)
{
    int i, j, k;
    double a;
    for (i = 0; i < N; i++)
        for (k = 0; k < N; k++) {
            a = A[i * N + k];
            for (j = 0; j < N; j++)
                C[i * N + j] += a * B[k * N + j];
        }
}

void mm_ikj_omp(const double *A, const double *B, double *C, int N)
{
    int i, j, k;
    double a;
    #pragma omp parallel for private(j, k, a)
    for (i = 0; i < N; i++)
        for (k = 0; k < N; k++) {
            a = A[i * N + k];
            for (j = 0; j < N; j++)
                C[i * N + j] += a * B[k * N + j];
        }
}

void mm_blocked_omp(const double *A, const double *B, double *C, int N, int b)
{
    int ii, jj, kk, i, j, k;
    double a;
    #pragma omp parallel for collapse(2) private(kk, i, j, k, a)
    for (ii = 0; ii < N; ii += b) {
        for (jj = 0; jj < N; jj += b) {
            for (kk = 0; kk < N; kk += b) {
                int i_max = (ii + b < N) ? ii + b : N;
                int j_max = (jj + b < N) ? jj + b : N;
                int k_max = (kk + b < N) ? kk + b : N;
                
                for (i = ii; i < i_max; i++) {
                    for (k = kk; k < k_max; k++) {
                        a = A[i * N + k];
                        for (j = jj; j < j_max; j++)
                            C[i * N + j] += a * B[k * N + j];
                    }
                }
            }
        }
    }
}

int run(const char *version, const double *A, const double *B, double *C, int N, int b)
{
    if (strcmp(version, "ijk_seq") == 0) mm_ijk_seq(A, B, C, N);
    else if (strcmp(version, "ijk_omp") == 0) mm_ijk_omp(A, B, C, N);
    else if (strcmp(version, "ikj_seq") == 0) mm_ikj_seq(A, B, C, N);
    else if (strcmp(version, "ikj_omp") == 0) mm_ikj_omp(A, B, C, N);
    else if (strcmp(version, "blocked_omp") == 0) mm_blocked_omp(A, B, C, N, b);
    else return 0;
    return 1;
}

void measure(const char *version, const double *A, const double *B, double *C,
             const double *Cref, int N, int b, int threads)
{
    double times[5], tmp, maxdiff, d;
    int r, p, q;
    long long k, size = (long long)N * N;
    int is_seq = (strcmp(version, "ijk_seq") == 0 || strcmp(version, "ikj_seq") == 0);

    // Прогревочный запуск
    memset(C, 0, size * sizeof(double));
    run(version, A, B, C, N, b);

    // 5 замеров
    for (r = 0; r < 5; r++) {
        memset(C, 0, size * sizeof(double)); 
        double start = omp_get_wtime();
        run(version, A, B, C, N, b);
        times[r] = omp_get_wtime() - start;
    }

    // Сортировка для медианы
    for (p = 0; p < 4; p++)
        for (q = 0; q < 4 - p; q++)
            if (times[q] > times[q + 1]) {
                tmp = times[q];
                times[q] = times[q + 1];
                times[q + 1] = tmp;
            }

    // Проверка точности
    maxdiff = 0.0;
    for (k = 0; k < size; k++) {
        d = fabs(C[k] - Cref[k]);
        if (d > maxdiff) maxdiff = d;
    }

    printf("N=%d b=%d threads=%d version=%-12s time=%.5f s maxdiff=%.2e check=%s\n",
           N, b, is_seq ? 1 : threads, version, times[2], maxdiff,
           maxdiff < 1e-6 ? "OK" : "FAIL");
}

int main(int argc, char *argv[])
{
    if (argc < 5) {
        printf("Usage: %s N b threads version\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int b = atoi(argv[2]);
    int threads = atoi(argv[3]);
    const char *version = argv[4];
    long long size = (long long)N * N;

    double *A = (double *)malloc(size * sizeof(double));
    double *B = (double *)malloc(size * sizeof(double));
    double *C = (double *)malloc(size * sizeof(double));
    double *Cref = (double *)malloc(size * sizeof(double));
    
    if (!A || !B || !C || !Cref) {
        printf("Memory allocation failed\n");
        return 1;
    }

    srand(1);
    for (long long k = 0; k < size; k++) {
        A[k] = rand() / (double)RAND_MAX;
        B[k] = rand() / (double)RAND_MAX;
    }

    omp_set_num_threads(threads);

    memset(Cref, 0, size * sizeof(double));
    mm_ijk_seq(A, B, Cref, N);

    if (strcmp(version, "all") == 0) {
        measure("ijk_seq", A, B, C, Cref, N, b, threads);
        measure("ijk_omp", A, B, C, Cref, N, b, threads);
        measure("ikj_seq", A, B, C, Cref, N, b, threads);
        measure("ikj_omp", A, B, C, Cref, N, b, threads);
        measure("blocked_omp", A, B, C, Cref, N, b, threads);
    } else {
        measure(version, A, B, C, Cref, N, b, threads);
    }

    free(A); free(B); free(C); free(Cref);
    return 0;
}