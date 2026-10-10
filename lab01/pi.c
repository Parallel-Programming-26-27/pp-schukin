#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char **argv) {
    long long n = 1000000000LL;          
    int threads = omp_get_max_threads(); 

    if (argc > 1) {
        n = atoll(argv[1]);
        if (n < 0) n = 0;
    }
    if (argc > 2) {
        threads = atoi(argv[2]);
        if (threads < 1) threads = 1;
    }

    omp_set_num_threads(threads);

    // Последовательная версия 
    double sum_seq = 0.0;
    double t0 = omp_get_wtime();

    for (long long i = 0; i < n; ++i) {
        double denom = (double)(2LL * i + 1LL);
        if (i & 1LL)
            sum_seq -= 1.0 / denom;
        else
            sum_seq += 1.0 / denom;
    }

    double t1 = omp_get_wtime();
    double time_seq = t1 - t0;

    double pi_seq = 4.0 * sum_seq;
    double err_seq = fabs(pi_seq - M_PI);

    // Параллельная версия OpenMP
    double sum_par = 0.0;
    t0 = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum_par) schedule(static) num_threads(threads)
    for (long long i = 0; i < n; ++i) {
        double denom = (double)(2LL * i + 1LL);
        if (i & 1LL)
            sum_par -= 1.0 / denom;
        else
            sum_par += 1.0 / denom;
    }

    t1 = omp_get_wtime();
    double time_par = t1 - t0;

    double pi_par = 4.0 * sum_par;
    double err_par = fabs(pi_par - M_PI);

    double speedup = (time_par > 0.0) ? (time_seq / time_par) : 0.0;

    // Вывод
    printf("n = %lld\n", n);
    printf("Threads = %d\n", threads);

    printf("\nSequential:\n");
    printf("  pi    = %.15f\n", pi_seq);
    printf("  error = %.6e\n", err_seq);
    printf("  time  = %.6f s\n", time_seq);

    printf("\nParallel OpenMP:\n");
    printf("  pi    = %.15f\n", pi_par);
    printf("  error = %.6e\n", err_par);
    printf("  time  = %.6f s\n", time_par);

    printf("\nSpeedup = %.6f\n", speedup);

    return 0;
}