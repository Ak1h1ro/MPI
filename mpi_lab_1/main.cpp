//15
/*#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int ProcRank, ProcNum;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

    printf("I am process %d from %d processes\n", ProcRank, ProcNum);

    MPI_Finalize();
    return 0;
}
*/

//16
/*#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    int ProcRank, ProcNum;
    // Инициализация MPI
    MPI_Init(&argc, &argv);
    // Получение количества процессов
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    // Получение ранга текущего процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    if (ProcRank == 0) {
        // Процесс с номером 0 выводит общее количество процессов
        std::cout << ProcNum << " processes." << std::endl;
    } else {
        // Остальные процессы выводят сообщения в зависимости от четности их номера
        if (ProcRank % 2 == 0) {
            // Процессы с четным номером выводят "SECOND!"
            std::cout << "I am " << ProcRank << ": SECOND!" << std::endl;
        } else {
            // Процессы с нечетным номером выводят "FIRST!"
            std::cout << "I am " << ProcRank << ": FIRST!" << std::endl;
        }
    }

    // Завершение работы с MPI
    MPI_Finalize();
    return 0;
}
*/

//17
/*#include <mpi.h>
#include <iostream>
#include <cstring>

int main(int argc, char** argv) {
    int ProcRank, ProcNum;
    // Инициализация MPI
    MPI_Init(&argc, &argv);
    // Получение количества процессов
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    // Получение ранга текущего процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    if (ProcNum < 2) {
        // Программа требует как минимум два процесса
        std::cerr << "Программа требует как минимум два процесса." << std::endl;
        MPI_Finalize();
        return 1;
    }
    if (ProcRank == 0) {
        // Процесс 0 отправляет сообщение процессу 1
        const char* message = "Hello from process 0!";
        MPI_Send(message, strlen(message) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    } else if (ProcRank == 1) {
        // Процесс 1 получает сообщение от процесса 0
        char received_message[100];
        MPI_Status status;
        MPI_Recv(received_message, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        // Вывод полученного сообщения
        std::cout << "Receive message: '" << received_message << "'" << std::endl;
    }
    // Завершение работы с MPI
    MPI_Finalize();
    return 0;
}
*/

//18
/*
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    int ProcRank, ProcNum;
    // Инициализация MPI
    MPI_Init(&argc, &argv);
    // Получение количества процессов
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    // Получение ранга текущего процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    int message = 0; // Переменная для хранения сообщения
    if (ProcRank == 0) {
        // Процесс 0 начинает передачу с отправки своего номера
        message = ProcRank;
        MPI_Send(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        std::cout << "Process " << ProcRank << " sent message: " << message << std::endl;
    } else {
        // Остальные процессы получают сообщение от предыдущего процесса
        MPI_Status status;
        MPI_Recv(&message, 1, MPI_INT, ProcRank - 1, 0, MPI_COMM_WORLD, &status);
        std::cout << "Process " << ProcRank << " received message: " << message << std::endl;
        // Инкрементируем сообщение
        message++;
        if (ProcRank < ProcNum - 1) {
            // Если это не последний процесс, отправляем сообщение следующему
            MPI_Send(&message, 1, MPI_INT, ProcRank + 1, 0, MPI_COMM_WORLD);
            std::cout << "Process " << ProcRank << " sent message: " << message << std::endl;
        } else {
            // Если это последний процесс, выводим окончательное значение
            std::cout << "Final message received by process " << ProcRank << ": " << message << std::endl;
        }
    }
    // Завершение работы с MPI
    MPI_Finalize();
    return 0;
}
*/

//19
/*
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    int ProcRank, ProcNum;
    // Инициализация MPI
    MPI_Init(&argc, &argv);
    // Получение количества процессов
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    // Получение ранга текущего процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    if (ProcRank == 0) {
        // Процесс 0 (master) принимает сообщения от всех остальных процессов
        for (int i = 1; i < ProcNum; i++) {
            int received_message;
            MPI_Status status;
            MPI_Recv(&received_message, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            std::cout << "Receive message: '" << received_message << "' from process " << i << std::endl;
        }
    } else {
        // Остальные процессы (slave) отправляют свои номера процессу 0
        int message = ProcRank;
        MPI_Send(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // Завершение работы с MPI
    MPI_Finalize();
    return 0;
}
*/

//20
/*
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    int ProcRank, ProcNum;
    // Инициализация MPI
    MPI_Init(&argc, &argv);
    // Получение количества процессов
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    // Получение ранга текущего процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    if (ProcNum < 2) {
        // Программа требует как минимум два процесса
        std::cerr << "Программа требует как минимум два процесса." << std::endl;
        MPI_Finalize();
        return 1;
    }
    if (ProcRank == 0) {
        // Процесс 0 отправляет сообщение процессу 1
        int message = 42; // Сообщение для отправки
        MPI_Request send_request;
        MPI_Isend(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &send_request);

        // Ожидание завершения отправки
        MPI_Status send_status;
        MPI_Wait(&send_request, &send_status);
        std::cout << "Process 0 sent message: " << message << std::endl;
    } else if (ProcRank == 1) {
        // Процесс 1 получает сообщение от процесса 0
        int received_message;
        MPI_Request recv_request;
        MPI_Irecv(&received_message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &recv_request);

        // Ожидание завершения приема
        MPI_Status recv_status;
        MPI_Wait(&recv_request, &recv_status);
        std::cout << "Process 1 received message: " << received_message << std::endl;
    }

    // Завершение работы с MPI
    MPI_Finalize();
    return 0;
}
*/

//21
/*
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    int ProcRank, ProcNum;

    // Инициализация MPI
    MPI_Init(&argc, &argv);

    // Получение количества процессов
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

    // Получение ранга текущего процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    int send_message = ProcRank; // Сообщение для отправки
    int recv_message;            // Буфер для приема сообщения

    // Определение рангов для отправки и приема
    int send_to = (ProcRank + 1) % ProcNum;       // Следующий процесс
    int recv_from = (ProcRank - 1 + ProcNum) % ProcNum; // Предыдущий процесс

    // Одновременная отправка и прием сообщения
    MPI_Sendrecv(&send_message, 1, MPI_INT, send_to, 0,
                 &recv_message, 1, MPI_INT, recv_from, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Вывод результата
    std::cout << "Process " << ProcRank << " received message: " << recv_message << std::endl;

    // Завершение работы с MPI
    MPI_Finalize();
    return 0;
}
*/

//22
/*
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    int ProcRank, ProcNum;

    // Инициализация MPI
    MPI_Init(&argc, &argv);

    // Получение количества процессов
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

    // Получение ранга текущего процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    // Каждый процесс отправляет свой номер всем остальным процессам
    for (int dest = 0; dest < ProcNum; dest++) {
        if (dest != ProcRank) {
            MPI_Send(&ProcRank, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }
    }

    // Каждый процесс получает сообщения от всех остальных процессов
    for (int src = 0; src < ProcNum; src++) {
        if (src != ProcRank) {
            int received_message;
            MPI_Recv(&received_message, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Вывод полученного сообщения
            std::cout << "[" << ProcRank << "]: receive message '" << received_message << "' from " << src << std::endl;
        }
    }
    // Вывод собственного сообщения
    std::cout << "[" << ProcRank << "]: receive message '" << ProcRank << "' from " << ProcRank << std::endl;

    // Завершение работы с MPI
    MPI_Finalize();
    return 0;
}
*/

//23
/*
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    std::string input_string;
    std::vector<int> global_counts(26, 0); // Для хранения общего количества каждой буквы

    if (rank == 0) {
        // Ввод данных
        std::cout << "Введите длину строки (1 <= n <= 100): ";
        std::cin >> n;
        std::cout << "Введите строку из " << n << " строчных английских букв: ";
        std::cin >> input_string;
    }

    // Рассылка длины строки всем процессам
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Рассылка строки всем процессам
    if (rank != 0) {
        input_string.resize(n);
    }
    MPI_Bcast(&input_string[0], n, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Подсчет вхождений символов в каждом процессе
    std::vector<int> local_counts(26, 0);
    int start = rank * n / size;
    int end = (rank + 1) * n / size;

    for (int i = start; i < end; ++i) {
        if (input_string[i] >= 'a' && input_string[i] <= 'z') {
            local_counts[input_string[i] - 'a']++;
        }
    }

    // Сбор результатов от всех процессов
    MPI_Reduce(local_counts.data(), global_counts.data(), 26, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Вывод результатов
    if (rank == 0) {
        for (int i = 0; i < 26; ++i) {
            if (global_counts[i] > 0) {
                std::cout << static_cast<char>('a' + i) << " = " << global_counts[i] << std::endl;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
*/

//23.2
/*
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    std::string input_string;
    std::vector<int> global_counts(26, 0);

    if (rank == 0) {
        // Ввод данных
        std::cout << "Введите длину строки (1 <= n <= 100): ";
        std::cin >> n;
        std::cout << "Введите строку из " << n << " строчных английских букв: ";
        std::cin >> input_string;

        // Отправка длины строки и строки каждому процессу
        for (int i = 1; i < size; ++i) {
            MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(input_string.data(), n, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
    } else {
        // Получение длины строки и строки
        MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        input_string.resize(n);
        MPI_Recv(input_string.data(), n, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Подсчет вхождений символов в каждом процессе
    std::vector<int> local_counts(26, 0);
    int start = rank * n / size;
    int end = (rank + 1) * n / size;

    for (int i = start; i < end; ++i) {
        if (input_string[i] >= 'a' && input_string[i] <= 'z') {
            local_counts[input_string[i] - 'a']++;
        }
    }

    // Сбор результатов от всех процессов
    MPI_Reduce(local_counts.data(), global_counts.data(), 26, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Вывод результатов
    if (rank == 0) {
        for (int i = 0; i < 26; ++i) {
            if (global_counts[i] > 0) {
                std::cout << static_cast<char>('a' + i) << " = " << global_counts[i] << std::endl;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
*/

//24
/*
#include <iostream>
#include <iomanip>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N;
    if (rank == 0) {
        std::cout << "Введите точность вычисления (N): ";
        std::cin >> N;
    }

    // Рассылаем значение N всем процессам
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double local_pi = 0.0;
    const double step = 1.0 / N;

    // Каждый процесс вычисляет свою часть суммы
    int start = rank * N / size;
    int end = (rank + 1) * N / size;

    for (int i = start; i < end; ++i) {
        double x = (i + 0.5) * step;
        local_pi += 4.0 / (1.0 + x * x);
    }

    local_pi *= step;

    // Собираем частичные суммы с помощью MPI_Reduce
    double global_pi;
    MPI_Reduce(&local_pi, &global_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Вывод результата в процессе 0
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(8) << global_pi << std::endl;
    }

    MPI_Finalize();
    return 0;
}
*/

//24.2
/*
#include <iostream>
#include <mpi.h>
#include <iomanip>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N;
    if (rank == 0) {
        std::cout << "Введите точность вычисления (N): ";
        std::cin >> N;
    }

    // Рассылаем значение N всем процессам
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double local_pi = 0.0;
    const double step = 1.0 / N;

    // Каждый процесс вычисляет свою часть суммы
    int start = rank * N / size;
    int end = (rank + 1) * N / size;

    for (int i = start; i < end; ++i) {
        double x = (i + 0.5) * step;
        local_pi += 4.0 / (1.0 + x * x);
    }

    local_pi *= step;

    // Сбор результатов с использованием MPI_Send и MPI_Recv
    if (rank != 0) {
        // Процессы отправляют свои результаты процессу 0
        MPI_Send(&local_pi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        // Процесс 0 собирает результаты от всех процессов
        double global_pi = local_pi;
        for (int i = 1; i < size; ++i) {
            double received_pi;
            MPI_Recv(&received_pi, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_pi += received_pi;
        }

        // Вывод результата
        std::cout << std::fixed << std::setprecision(8) << global_pi << std::endl;
    }

    MPI_Finalize();
    return 0;
}
*/

//25
/*
#include <iostream>
#include <mpi.h>
#include <vector>
#include <iomanip>

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n, int rows, int offset) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; ++k) {
                C[i * n + j] += A[(i + offset) * n + k] * B[k * n + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    std::vector<double> A, B, C;

    if (rank == 0) {
        std::cout << "Введите размер матрицы (n x n): ";
        std::cin >> n;

        if (n <= 0) {
            std::cerr << "Ошибка: размер матрицы должен быть положительным!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        A.resize(n * n);
        B.resize(n * n);
        C.resize(n * n);

        std::cout << "Введите элементы матрицы A:" << std::endl;
        for (int i = 0; i < n * n; ++i) {
            std::cin >> A[i];
        }

        std::cout << "Введите элементы матрицы B:" << std::endl;
        for (int i = 0; i < n * n; ++i) {
            std::cin >> B[i];
        }
    }

    // Рассылаем размер матрицы всем процессам
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n <= 0) {
        MPI_Finalize();
        return 1;
    }

    // Вычисляем, сколько строк должен обработать текущий процесс
    int rows_per_process = n / size;
    int remainder = n % size;

    int local_rows = (rank < remainder) ? rows_per_process + 1 : rows_per_process;
    int offset = (rank < remainder)
        ? rank * (rows_per_process + 1)
        : remainder * (rows_per_process + 1) + (rank - remainder) * rows_per_process;

    // Если процессу не досталось строк, завершаем его работу
    if (local_rows <= 0) {
        MPI_Finalize();
        return 0;
    }

    // Выделяем память под локальные данные
    std::vector<double> local_A(local_rows * n);
    std::vector<double> local_C(local_rows * n);

    // Рассылаем матрицу B всем процессам
    if (rank != 0) {
        B.resize(n * n);
    }
    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Разделяем матрицу A с помощью MPI_Scatterv (только если процессу достались строки)
    if (local_rows > 0) {
        std::vector<int> sendcounts(size);
        std::vector<int> displs(size);

        for (int i = 0; i < size; ++i) {
            sendcounts[i] = (i < remainder) ? (rows_per_process + 1) * n : rows_per_process * n;
            displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
        }

        MPI_Scatterv(
            A.data(),
            sendcounts.data(),
            displs.data(),
            MPI_DOUBLE,
            local_A.data(),
            local_rows * n,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        // Вычисляем произведение
        matrixMultiply(local_A, B, local_C, n, local_rows, 0);

        // Собираем результаты
        MPI_Gatherv(
            local_C.data(),
            local_rows * n,
            MPI_DOUBLE,
            C.data(),
            sendcounts.data(),
            displs.data(),
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );
    }

    // Вывод результата (только root-процесс)
    if (rank == 0) {
        std::cout << "Результирующая матрица C:" << std::endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cout << std::setw(10) << C[i * n + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
*/

//25.2
/*
#include <iostream>
#include <mpi.h>
#include <vector>
#include <iomanip>

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, int n, int rows, int offset) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; ++k) {
                C[i * n + j] += A[(i + offset) * n + k] * B[k * n + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    std::vector<double> A, B, C;

    // Ввод данных в процессе 0
    if (rank == 0) {
        std::cout << "Введите размер матрицы (n x n): ";
        std::cin >> n;

        if (n <= 0) {
            std::cerr << "Ошибка: размер матрицы должен быть положительным!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        A.resize(n * n);
        B.resize(n * n);
        C.resize(n * n);

        std::cout << "Введите элементы матрицы A:" << std::endl;
        for (int i = 0; i < n * n; ++i) {
            std::cin >> A[i];
        }

        std::cout << "Введите элементы матрицы B:" << std::endl;
        for (int i = 0; i < n * n; ++i) {
            std::cin >> B[i];
        }
    }

    // Рассылка размера матрицы всем процессам
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n <= 0) {
        MPI_Finalize();
        return 1;
    }

    // Вычисление количества строк для каждого процесса
    int rows_per_process = n / size;
    int remainder = n % size;

    int local_rows = (rank < remainder) ? rows_per_process + 1 : rows_per_process;
    int offset = (rank < remainder)
                 ? rank * (rows_per_process + 1)
                 : remainder * (rows_per_process + 1) + (rank - remainder) * rows_per_process;

    // Выделение памяти под локальные данные
    std::vector<double> local_A(local_rows * n);
    std::vector<double> local_C(local_rows * n);

    // Рассылка матрицы B всем процессам
    if (rank != 0) {
        B.resize(n * n);
    }
    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Распределение строк матрицы A с помощью MPI_Send/MPI_Recv
    if (rank == 0) {
        // Отправка данных другим процессам
        for (int dest = 1; dest < size; ++dest) {
            int dest_rows = (dest < remainder) ? rows_per_process + 1 : rows_per_process;
            int dest_offset = (dest < remainder)
                             ? dest * (rows_per_process + 1)
                             : remainder * (rows_per_process + 1) + (dest - remainder) * rows_per_process;

            if (dest_rows > 0) {
                MPI_Send(&A[dest_offset * n], dest_rows * n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }
        }
        // Копирование своих данных
        if (local_rows > 0) {
            std::copy(A.begin() + offset * n,
                     A.begin() + (offset + local_rows) * n,
                     local_A.begin());
        }
    } else {
        // Получение данных от процесса 0
        if (local_rows > 0) {
            MPI_Recv(local_A.data(), local_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Вычисление локальной части произведения
    if (local_rows > 0) {
        matrixMultiply(local_A, B, local_C, n, local_rows, 0);
    }

    // Сбор результатов с помощью MPI_Send/MPI_Recv
    if (rank == 0) {
        // Копирование своих результатов
        if (local_rows > 0) {
            std::copy(local_C.begin(),
                     local_C.end(),
                     C.begin() + offset * n);
        }

        // Получение результатов от других процессов
        for (int src = 1; src < size; ++src) {
            int src_rows = (src < remainder) ? rows_per_process + 1 : rows_per_process;
            int src_offset = (src < remainder)
                            ? src * (rows_per_process + 1)
                            : remainder * (rows_per_process + 1) + (src - remainder) * rows_per_process;

            if (src_rows > 0) {
                MPI_Recv(&C[src_offset * n], src_rows * n, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        // Отправка результатов процессу 0
        if (local_rows > 0) {
            MPI_Send(local_C.data(), local_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    // Вывод результата
    if (rank == 0) {
        std::cout << "Результирующая матрица C:" << std::endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cout << std::setw(10) << C[i * n + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
*/

//26
/*
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Чтение сообщения процессом 0
    char message[11] = {0}; // Максимум 10 символов + '\0'
    if (world_rank == 0) {
        std::cout << "Enter message (1-10 characters): ";
        std::cin >> message;

        // Проверка длины сообщения
        if (strlen(message) < 1 || strlen(message) > 10) {
            std::cerr << "Error: message length must be 1-10 characters" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Создаем группу для процессов с четными номерами
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    // Выбираем процессы с четными номерами
    std::vector<int> even_ranks;
    for (int i = 0; i < world_size; i += 2) {
        even_ranks.push_back(i);
    }

    // Создаем новую группу
    MPI_Group even_group;
    MPI_Group_incl(world_group, even_ranks.size(), even_ranks.data(), &even_group);

    // Создаем новый коммуникатор
    MPI_Comm even_comm;
    MPI_Comm_create(MPI_COMM_WORLD, even_group, &even_comm);
    // Получаем информацию о новом коммуникаторе
    int even_rank = MPI_UNDEFINED, even_size = 0;
    if (even_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(even_comm, &even_rank);
        MPI_Comm_size(even_comm, &even_size);
        // Рассылаем сообщение только в новом коммуникаторе
        MPI_Bcast(message, 10, MPI_CHAR, 0, even_comm);
    }

    // Синхронизация перед выводом
    MPI_Barrier(MPI_COMM_WORLD);

    // Вывод информации
    std::cout << "MPI_COMM_WORLD: " << world_rank << " from " << world_size
              << ". New comm: " << (even_rank == MPI_UNDEFINED ? -1 : even_rank)
              << " from " << even_size << ". Message = ";
    if (even_comm != MPI_COMM_NULL) {
        std::cout << message;
    } else {
        std::cout << "no";
    }
    std::cout << std::endl;
    // Освобождение ресурсов
    if (even_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&even_comm);
    }
    MPI_Group_free(&even_group);
    MPI_Group_free(&world_group);
    MPI_Finalize();
    return 0;
}
*/

//27
/*
#include <iostream>
#include <mpi.h>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm parent_comm;
    MPI_Comm_get_parent(&parent_comm);

    if (parent_comm == MPI_COMM_NULL) {
        // Родительский процесс
        if (world_rank == 0) {
            int n;
            std::cout << "Enter number of child processes to spawn: ";
            std::cin >> n;

            std::cout << "I am " << world_rank << " process from "
                      << world_size << " processes!\n"
                      << "My parent is none." << std::endl;

            // Создаем дочерние процессы
            MPI_Comm intercomm;
            MPI_Comm_spawn(argv[0], MPI_ARGV_NULL, n, MPI_INFO_NULL,
                         0, MPI_COMM_WORLD, &intercomm, MPI_ERRCODES_IGNORE);

            // Обмен данными с дочерними процессами
            int data_to_send = 100;
            MPI_Bcast(&data_to_send, 1, MPI_INT, MPI_ROOT, intercomm);

            // Собираем результаты от дочерних процессов
            if (n > 0) {
                std::vector<int> results(n);
                MPI_Comm child_comm;
                MPI_Intercomm_merge(intercomm, 0, &child_comm);

                MPI_Gather(MPI_IN_PLACE, 0, MPI_INT,
                          results.data(), 1, MPI_INT,
                          0, child_comm);

                std::cout << "Results from children:";
                for (int res : results) {
                    std::cout << " " << res;
                }
                std::cout << std::endl;

                MPI_Comm_free(&child_comm);
            }

            MPI_Comm_free(&intercomm);
        }
    } else {
        // Дочерний процесс
        int child_rank, child_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &child_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &child_size);

        std::cout << "I am " << child_rank << " process from "
                  << child_size << " processes!\n"
                  << "My parent is 0." << std::endl;

        // Получаем данные от родителя
        int received_data;
        MPI_Bcast(&received_data, 1, MPI_INT, 0, parent_comm);

        // Вычисляем результат
        int result = received_data + child_rank;

        // Отправляем результат родителю
        MPI_Comm child_comm;
        MPI_Intercomm_merge(parent_comm, 1, &child_comm);

        MPI_Gather(&result, 1, MPI_INT,
                  nullptr, 0, MPI_INT,
                  0, child_comm);

        MPI_Comm_free(&child_comm);
        MPI_Comm_free(&parent_comm);
    }

    MPI_Finalize();
    return 0;
}
*/

//28
/*
#include <iostream>
#include <iomanip>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N;
    if (rank == 0) {
        std::cout << "Введите точность вычисления (N): ";
        std::cin >> N;
    }

    // Рассылаем значение N всем процессам
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double local_pi = 0.0;
    const double step = 1.0 / N;

    // Каждый процесс вычисляет свою часть суммы
    int start = rank * N / size;
    int end = (rank + 1) * N / size;

    for (int i = start; i < end; ++i) {
        double x = (i + 0.5) * step;
        local_pi += 4.0 / (1.0 + x * x);
    }
    local_pi *= step;

    // Создаем окно для односторонней коммуникации
    double global_pi = 0.0;
    MPI_Win win;
    if (rank == 0) {
        // Процесс 0 выделяет память для результата
        MPI_Win_create(&global_pi, sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    } else {
        // Остальные процессы не выделяют память
        MPI_Win_create(nullptr, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }

    // Начинаем доступ к окну
    MPI_Win_fence(0, win);

    if (rank != 0) {
        // Процессы, кроме 0, добавляют свой локальный результат в global_pi процесса 0
        MPI_Accumulate(&local_pi, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE, MPI_SUM, win);
    } else {
        // Процесс 0 добавляет свой локальный результат напрямую
        global_pi += local_pi;
    }

    // Завершаем доступ к окну
    MPI_Win_fence(0, win);

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(8) << global_pi << std::endl;
    }

    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
*/

//29
/*
#include <iostream>
#include <iomanip>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N;
    if (rank == 0) {
        std::cout << "Введите точность вычисления (N): ";
        std::cin >> N;
    }

    // Рассылаем значение N всем процессам
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Засекаем время начала вычислений
    double start_time = MPI_Wtime();

    double local_pi = 0.0;
    const double step = 1.0 / N;

    // Каждый процесс вычисляет свою часть суммы
    int start = rank * N / size;
    int end = (rank + 1) * N / size;

    for (int i = start; i < end; ++i) {
        double x = (i + 0.5) * step;
        local_pi += 4.0 / (1.0 + x * x);
    }

    local_pi *= step;

    // Собираем частичные суммы с помощью MPI_Reduce
    double global_pi;
    MPI_Reduce(&local_pi, &global_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Засекаем время окончания вычислений
    double end_time = MPI_Wtime();

    // Вывод результата и времени выполнения в процессе 0
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "Pi = " << global_pi << std::endl;
        std::cout << "Время выполнения: " << (end_time - start_time) << " секунд" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
*/

//30
/*
#include <iostream>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nthreads = 0;
    #pragma omp parallel
    {
        #pragma omp master
        nthreads = omp_get_num_threads();
    }

    std::cout << "Hello from MPI process " << rank << " out of " << size
              << " processes, running with " << nthreads << " OpenMP threads." << std::endl;

    MPI_Finalize();
    return 0;
}
*/

//31
/*
#include <iostream>
#include <mpi.h>
#include <omp.h>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n;
    if (world_rank == 0) {
        std::cout << "Введите количество нитей: ";
        std::cin >> n;
    }

    // Рассылаем значение n всем процессам
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_threads = n * world_size;
    #pragma omp parallel num_threads(n)
    {
        int thread_num = omp_get_thread_num();
        // Печатаем требуемую строку
        printf("I am %d thread from %d process. Number of hybrid threads = %d\n",
               thread_num, world_rank, total_threads);
    }
    MPI_Finalize();
    return 0;
}
*/

//32
/*
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N;
    if (rank == 0) {
        std::cout << "Введите точность вычисления (N): ";
        std::cin >> N;
    }

    // Рассылаем значение N всем процессам
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double step = 1.0 / N;
    double local_sum = 0.0;

    // Определяем границы работы для каждого процесса
    int start = rank * N / size;
    int end = (rank + 1) * N / size;

    // Параллельный цикл OpenMP внутри процесса
    #pragma omp parallel for reduction(+:local_sum)
    for (int i = start; i < end; ++i) {
        double x = (i + 0.5) * step;
        local_sum += 4.0 / (1.0 + x * x);
    }

    local_sum *= step;

    // Суммируем результаты всех процессов с помощью MPI_Reduce
    double global_pi = 0.0;
    MPI_Reduce(&local_sum, &global_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(8) << global_pi << std::endl;
    }

    MPI_Finalize();
    return 0;
}
*/
