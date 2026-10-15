#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

// Вариант F: f(x) = x^3 * e^(2x)
static double f(double x) {
    return x * x * x * exp(2.0 * x);
}

// Последовательное вычисление интеграла методом средних прямоугольников
static double integrate_serial(double a, double b, long long n) {
    double h = (b - a) / (double)n;
    double sum = 0.0;

    for (long long i = 0; i < n; ++i) {
        double x = a + ((double)i + 0.5) * h;
        sum += f(x);
    }

    return sum * h;
}

// Параллельное вычисление интеграла с OpenMP
static double integrate_parallel(double a, double b, long long n, int threads) {
    double h = (b - a) / (double)n;
    double sum = 0.0;

    #pragma omp parallel for num_threads(threads) reduction(+:sum) schedule(static)
    for (long long i = 0; i < n; ++i) {
        double x = a + ((double)i + 0.5) * h;
        sum += f(x);
    }

    return sum * h;
}

// Эксперимент 1: сравнение времени при 1, 2, 4 и 8 потоках
static void experiment_threads(double a, double b) {
    const long long n = 100000000LL;
    const int counts[] = {1, 2, 4, 8};

    double start = omp_get_wtime();
    double serial = integrate_serial(a, b, n);
    double serial_time = omp_get_wtime() - start;

    printf("\nEXPERIMENT 1: n = %lld\n", n);
    printf("Serial: I = %.12f; time = %.6f s\n", serial, serial_time);
    printf("Threads     Integral           Time (s)     Speedup vs serial\n");

    for (int i = 0; i < 4; ++i) {
        int threads = counts[i];

        start = omp_get_wtime();
        double result = integrate_parallel(a, b, n, threads);
        double elapsed = omp_get_wtime() - start;

        double speedup = 0.0;
        if (elapsed > 0.0) {
            speedup = serial_time / elapsed;
        }

        printf("%-11d %.12f     %.6f     %.3f\n",
               threads, result, elapsed, speedup);
    }
}

// Эксперимент 2: влияние числа разбиений при 8 потоках
static void experiment_partitions(double a, double b) {
    const long long counts[] = {
        10000LL, 100000LL, 1000000LL, 10000000LL, 100000000LL
    };
    const int threads = 8;

    printf("\nEXPERIMENT 2: threads = %d\n", threads);
    printf("Partitions        Integral           Time (s)\n");

    for (int i = 0; i < 5; ++i) {
        long long n = counts[i];

        double start = omp_get_wtime();
        double result = integrate_parallel(a, b, n, threads);
        double elapsed = omp_get_wtime() - start;

        printf("%-17lld %.12f     %.6f\n", n, result, elapsed);
    }
}

// Эксперимент 3: подбор числа разбиений для заданной точности
static void experiment_accuracy(double a, double b, double eps, int threads) {
    long long n = 10;
    double previous = integrate_serial(a, b, n);

    printf("\nEXPERIMENT 3: tolerance = %.3g; threads = %d\n", eps, threads);

    while (n <= 1000000000LL / 2) {
        n *= 2;

        // Время последовательной версии
        double start = omp_get_wtime();
        double serial = integrate_serial(a, b, n);
        double serial_time = omp_get_wtime() - start;

        // Время параллельной версии
        start = omp_get_wtime();
        double parallel = integrate_parallel(a, b, n, threads);
        double parallel_time = omp_get_wtime() - start;

        // Сравнение результатов при n и n/2
        double difference = fabs(serial - previous);

        if (difference < eps) {
            printf("n = %lld; I = %.12f; |I(n)-I(n/2)| = %.12g\n",
                   n, serial, difference);
            printf("Serial time = %.6f s; parallel time = %.6f s\n",
                   serial_time, parallel_time);
            printf("Parallel result = %.12f\n", parallel);
            return;
        }

        previous = serial;
    }

    printf("Tolerance not reached before partition limit.\n");
}

// Подсказка по аргументам командной строки
static void usage(const char *name) {
    printf("Usage:\n");
    printf("  %s                      Run all experiments with variant F defaults\n", name);
    printf("  %s --all [eps]          Run all experiments; default eps=1e-8\n", name);
    printf("  %s --single a b n threads  Calculate integral once\n", name);
    printf("  %s --accuracy [eps] [threads]  Find partitions for tolerance\n", name);
    printf("  %s --help               Show this help\n", name);
}

int main(int argc, char **argv) {
    const double a = 0.0;
    const double b = 1.0;

    omp_set_dynamic(0);

    // Обычный запуск или все три эксперимента
    if (argc == 1 || (argc >= 2 && strcmp(argv[1], "--all") == 0)) {
        double eps = 1e-8;
        if (argc >= 3) {
            eps = atof(argv[2]);
        }

        if (eps <= 0.0 || !isfinite(eps)) {
            fprintf(stderr, "Invalid tolerance\n");
            return 1;
        }

        printf("Variant F: f(x) = x^3 * exp(2*x), interval [0, 1]\n");
        printf("Exact integral = %.12f\n", (exp(2.0) + 3.0) / 8.0);

        experiment_threads(a, b);
        experiment_partitions(a, b);
        experiment_accuracy(a, b, eps, 8);
    }

    // Одно вычисление с заданными границами, n и числом потоков
    else if (strcmp(argv[1], "--single") == 0 && argc == 6) {
        char *end;

        double left = strtod(argv[2], &end);
        if (*end || !isfinite(left)) {
            fprintf(stderr, "Invalid a\n");
            return 1;
        }

        double right = strtod(argv[3], &end);
        if (*end || !isfinite(right) || right <= left) {
            fprintf(stderr, "Invalid b\n");
            return 1;
        }

        long long n = strtoll(argv[4], &end, 10);
        if (*end || n <= 0) {
            fprintf(stderr, "Invalid n\n");
            return 1;
        }

        long threads_long = strtol(argv[5], &end, 10);
        if (*end || threads_long < 1 || threads_long > 8) {
            fprintf(stderr, "Invalid threads (must be 1..8)\n");
            return 1;
        }
        int threads = (int)threads_long;

        // Последовательная версия
        double start = omp_get_wtime();
        double serial = integrate_serial(left, right, n);
        double serial_time = omp_get_wtime() - start;

        // Параллельная версия
        start = omp_get_wtime();
        double parallel = integrate_parallel(left, right, n, threads);
        double parallel_time = omp_get_wtime() - start;

        double speedup = 0.0;
        if (parallel_time > 0.0) {
            speedup = serial_time / parallel_time;
        }

        // Вывод результата
        printf("Interval [%.6g, %.6g], n=%lld, threads=%d\n", left, right, n, threads);
        printf("Serial:   I = %.12f, time = %.6f s\n", serial, serial_time);
        printf("Parallel: I = %.12f, time = %.6f s\n", parallel, parallel_time);
        printf("Speedup = %.3f\n", speedup);
    }

    // Вычисление с заданной точностью
    else if (strcmp(argv[1], "--accuracy") == 0 && argc <= 4) {
        double eps = 1e-8;
        int threads = 8;

        if (argc >= 3) {
            eps = atof(argv[2]);
        }
        if (argc >= 4) {
            threads = atoi(argv[3]);
        }

        if (eps <= 0.0 || !isfinite(eps) || threads < 1 || threads > 8) {
            fprintf(stderr, "Invalid accuracy or threads (1..8)\n");
            return 1;
        }

        experiment_accuracy(a, b, eps, threads);
    }

    // Помощь или неизвестная команда
    else {
        usage(argv[0]);

        if (argc == 2 && strcmp(argv[1], "--help") == 0) {
            return 0;
        }
        return 1;
    }

    return 0;
}
