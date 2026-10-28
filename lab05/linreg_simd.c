#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// Оператор мягкого порогового отсечения
double soft_threshold(double z, double alpha) {
    if (z > alpha) return z - alpha;
    if (z < -alpha) return z + alpha;
    return 0.0;
}

// Вычисление функции потерь F(w,b)
double compute_loss(const double *X, const double *y, const double *w, double b, int N, int d, double lambda) {
    double sum_sq = 0.0;
    double sum_abs_w = 0.0;
    
    for (int i = 0; i < N; i++) {
        double pred = b;
        for (int j = 0; j < d; j++) {
            pred += X[i * d + j] * w[j];
        }
        double err = pred - y[i];
        sum_sq += err * err;
    }
    for (int j = 0; j < d; j++) {
        sum_abs_w += fabs(w[j]);
    }
    
    return (0.5 / N) * sum_sq + lambda * sum_abs_w;
}

// 1. Последовательная версия
void train_serial(const double *X, const double *y, double *w, double *b, double *e, double *g, 
                  int N, int d, int K, double lambda, double eta) {
    for (int k = 0; k < K; k++) {
        // Вычисление ошибок e_i
        for (int i = 0; i < N; i++) {
            double pred = 0.0;
            for (int j = 0; j < d; j++) pred += X[i * d + j] * w[j];
            e[i] = pred + *b - y[i];
        }
        
        // Вычисление градиентов
        double gb = 0.0;
        for (int j = 0; j < d; j++) g[j] = 0.0;
        
        for (int i = 0; i < N; i++) {
            gb += e[i];
            for (int j = 0; j < d; j++) g[j] += e[i] * X[i * d + j];
        }
        
        // Обновление b и w
        *b -= eta * (gb / N);
        for (int j = 0; j < d; j++) {
            g[j] /= N;
            double z = w[j] - eta * g[j];
            w[j] = soft_threshold(z, eta * lambda);
        }
    }
}

// 2. Версия с OpenMP
void train_omp(const double *X, const double *y, double *w, double *b, double *e, double *g, 
               int N, int d, int K, double lambda, double eta) {
    int max_threads = omp_get_max_threads();
    double *g_local = (double *)calloc(max_threads * d, sizeof(double));
    
    for (int k = 0; k < K; k++) {
        double gb = 0.0;
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double *my_g = &g_local[tid * d];
            for (int j = 0; j < d; j++) my_g[j] = 0.0;

            // Вычисление ошибок
            #pragma omp for reduction(+:gb)
            for (int i = 0; i < N; i++) {
                double pred = 0.0;
                for (int j = 0; j < d; j++) pred += X[i * d + j] * w[j];
                e[i] = pred + *b - y[i];
                gb += e[i];
            }

            // Локальное накопление градиентов
            #pragma omp for
            for (int i = 0; i < N; i++) {
                double ei = e[i];
                for (int j = 0; j < d; j++) my_g[j] += ei * X[i * d + j];
            }

            // Слияние локальных градиентов в глобальный
            #pragma omp for
            for (int j = 0; j < d; j++) {
                double sum = 0.0;
                for (int t = 0; t < max_threads; t++) sum += g_local[t * d + j];
                g[j] = sum / N;
            }
        }
        
        *b -= eta * (gb / N);
        for (int j = 0; j < d; j++) {
            double z = w[j] - eta * g[j];
            w[j] = soft_threshold(z, eta * lambda);
        }
    }
    free(g_local);
}

// 3. Версия с SIMD
void train_simd(const double *X, const double *y, double *w, double *b, double *e, double *g, 
                int N, int d, int K, double lambda, double eta) {
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < N; i++) {
            double pred = 0.0;
            #pragma omp simd reduction(+:pred)
            for (int j = 0; j < d; j++) pred += X[i * d + j] * w[j];
            e[i] = pred + *b - y[i];
        }
        
        double gb = 0.0;
        for (int j = 0; j < d; j++) g[j] = 0.0;
        
        for (int i = 0; i < N; i++) {
            gb += e[i];
            double ei = e[i];
            #pragma omp simd
            for (int j = 0; j < d; j++) g[j] += ei * X[i * d + j];
        }
        
        *b -= eta * (gb / N);
        for (int j = 0; j < d; j++) {
            g[j] /= N;
            double z = w[j] - eta * g[j];
            w[j] = soft_threshold(z, eta * lambda);
        }
    }
}

// 4. Комбинированная версия OpenMP + SIMD
void train_omp_simd(const double *X, const double *y, double *w, double *b, double *e, double *g, 
                    int N, int d, int K, double lambda, double eta) {
    int max_threads = omp_get_max_threads();
    double *g_local = (double *)calloc(max_threads * d, sizeof(double));
    
    for (int k = 0; k < K; k++) {
        double gb = 0.0;
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double *my_g = &g_local[tid * d];
            for (int j = 0; j < d; j++) my_g[j] = 0.0;

            #pragma omp for reduction(+:gb)
            for (int i = 0; i < N; i++) {
                double pred = 0.0;
                #pragma omp simd reduction(+:pred)
                for (int j = 0; j < d; j++) pred += X[i * d + j] * w[j];
                e[i] = pred + *b - y[i];
                gb += e[i];
            }

            #pragma omp for
            for (int i = 0; i < N; i++) {
                double ei = e[i];
                #pragma omp simd
                for (int j = 0; j < d; j++) my_g[j] += ei * X[i * d + j];
            }

            #pragma omp for
            for (int j = 0; j < d; j++) {
                double sum = 0.0;
                for (int t = 0; t < max_threads; t++) sum += g_local[t * d + j];
                g[j] = sum / N;
            }
        }
        
        *b -= eta * (gb / N);
        for (int j = 0; j < d; j++) {
            double z = w[j] - eta * g[j];
            w[j] = soft_threshold(z, eta * lambda);
        }
    }
    free(g_local);
}

int main(int argc, char *argv[]) {
    if (argc < 8) {
        printf("Usage: %s N d K lambda eta threads version(s|omp|simd|omp_simd)\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int d = atoi(argv[2]);
    int K = atoi(argv[3]);
    double lambda = atof(argv[4]);
    double eta = atof(argv[5]);
    int threads = atoi(argv[6]);
    const char *version = argv[7];

    omp_set_num_threads(threads);

    // Выделение памяти
    double *X = (double *)malloc(N * d * sizeof(double));
    double *y = (double *)malloc(N * sizeof(double));
    double *w = (double *)calloc(d, sizeof(double));
    double *w_ref = (double *)calloc(d, sizeof(double));
    double *e = (double *)malloc(N * sizeof(double));
    double *g = (double *)malloc(d * sizeof(double));
    
    double b = 0.0, b_ref = 0.0;

    // Генерация данных
    srand(42);
    for (int i = 0; i < N * d; i++) X[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    
    // Истинные веса
    double *w_true = (double *)calloc(d, sizeof(double));
    for (int j = 0; j < d; j++) {
        if (j % 2 == 0) w_true[j] = 1.0; 
    }
    double b_true = 0.5;

    for (int i = 0; i < N; i++) {
        double pred = b_true;
        for (int j = 0; j < d; j++) pred += X[i * d + j] * w_true[j];
        y[i] = pred + ((double)rand() / RAND_MAX * 0.1 - 0.05); // небольшой шум
    }

    // Эталонный запуск
    train_serial(X, y, w_ref, &b_ref, e, g, N, d, K, lambda, eta);

    // Выбор версии для замера
    void (*train_func)(const double*, const double*, double*, double*, double*, double*, int, int, int, double, double) = NULL;
    if (strcmp(version, "s") == 0) train_func = train_serial;
    else if (strcmp(version, "omp") == 0) train_func = train_omp;
    else if (strcmp(version, "simd") == 0) train_func = train_simd;
    else if (strcmp(version, "omp_simd") == 0) train_func = train_omp_simd;
    else {
        printf("Unknown version: %s\n", version);
        return 1;
    }

    // Прогревочный запуск
    train_func(X, y, w, &b, e, g, N, d, K, lambda, eta);

    // Замеры
    double times[5];
    for (int r = 0; r < 5; r++) {
        for (int j = 0; j < d; j++) w[j] = 0.0;
        b = 0.0;
        double start = omp_get_wtime();
        train_func(X, y, w, &b, e, g, N, d, K, lambda, eta);
        times[r] = omp_get_wtime() - start;
    }

    // Сортировка для медианы
    for (int p = 0; p < 4; p++)
        for (int q = 0; q < 4 - p; q++)
            if (times[q] > times[q + 1]) {
                double tmp = times[q]; times[q] = times[q + 1]; times[q + 1] = tmp;
            }
    double med_time = times[2];

    // Проверка корректности
    double max_diff = 0.0;
    for (int j = 0; j < d; j++) {
        double diff = fabs(w[j] - w_ref[j]);
        if (diff > max_diff) max_diff = diff;
    }

    // Сбор статистики
    double final_loss = compute_loss(X, y, w, b, N, d, lambda);
    int zero_coeffs = 0;
    for (int j = 0; j < d; j++) if (fabs(w[j]) < 1e-6) zero_coeffs++;

    printf("N=%d d=%d K=%d lambda=%.4f eta=%.4f threads=%d version=%-8s time=%.5f s loss=%.4f maxdiff=%.2e zero_w=%d check=%s\n",
           N, d, K, lambda, eta, threads, version, med_time, final_loss, max_diff, zero_coeffs, 
           max_diff < 1e-4 ? "OK" : "FAIL");

    // Очистка
    free(X); free(y); free(w); free(w_ref); free(e); free(g); free(w_true);
    return 0;
}