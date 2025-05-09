#include <iostream>
#include <ctime>
#include <cstdlib>
#include <utility> // For std::swap
#include <omp.h>   // OpenMP header

using namespace std;

// Parallel Bubble Sort using OpenMP
void bubbleSort(int arr[], int size)
{
    for (int i = 0; i < size; i++)
    {
        bool swapped = false;

// Parallelize the inner loop (odd-even approach for better parallelism)
#pragma omp parallel for shared(arr, swapped)
        for (int j = 0; j < size - i - 1; j++)
        {
            if (arr[j + 1] < arr[j])
            {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }

        if (!swapped)
        {
            break; // Early termination if no swaps occurred
        }
    }
}

int main()
{
    const int SIZE = 1000;
    int arr[SIZE];

    // Initialize array with random values
    srand(time(0));
    for (int i = 0; i < SIZE; i++)
    {
        arr[i] = rand() % 1000;
    }

    // Display first and last elements
    cout << "Original array: [";
    for (int i = 0; i < 5; i++)
    {
        cout << arr[i] << " ";
    }
    cout << "............ ";
    for (int i = SIZE - 5; i < SIZE; i++)
    {
        cout << arr[i] << " ";
    }
    cout << "\b]\n";

    // Sort and measure time
    clock_t start = clock();
    bubbleSort(arr, SIZE);
    clock_t end = clock();

    // Display sorted results
    cout << "Sorted array: [";
    for (int i = 0; i < 5; i++)
    {
        cout << arr[i] << " ";
    }
    cout << "... ";
    for (int i = SIZE - 5; i < SIZE; i++)
    {
        cout << arr[i] << " ";
    }
    cout << "\b]\n";

    double time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout << "Execution time: " << time_taken << " seconds\n";

    return 0;
}