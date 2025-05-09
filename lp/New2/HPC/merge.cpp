#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <omp.h> // OpenMP header

using namespace std;

vector<int> merge(vector<int> &left, vector<int> &right)
{
  vector<int> result;
  int i = 0, j = 0;

  while (i < left.size() && j < right.size())
  {
    if (left[i] <= right[j])
    {
      result.push_back(left[i++]);
    }
    else
    {
      result.push_back(right[j++]);
    }
  }

  while (i < left.size())
  {
    result.push_back(left[i++]);
  }

  while (j < right.size())
  {
    result.push_back(right[j++]);
  }

  return result;
}

vector<int> mergeSort(vector<int> &arr)
{
  if (arr.size() <= 1)
  {
    return arr;
  }

  int mid = arr.size() / 2;
  vector<int> left(arr.begin(), arr.begin() + mid);
  vector<int> right(arr.begin() + mid, arr.end());

// Parallelize the recursive calls
#pragma omp parallel sections
  {
#pragma omp section
    {
      left = mergeSort(left);
    }
#pragma omp section
    {
      right = mergeSort(right);
    }
  }

  return merge(left, right);
}

int main()
{
  const int SIZE = 100000; // Increased size to better see parallelization benefits
  vector<int> arr(SIZE);

  srand(time(0));
  for (int i = 0; i < SIZE; i++)
  {
    arr[i] = rand() % 10000;
  }

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

  clock_t start = clock();
  vector<int> sorted = mergeSort(arr);
  clock_t end = clock();

  cout << "Sorted array: [";
  for (int i = 0; i < 5; i++)
  {
    cout << sorted[i] << " ";
  }
  cout << "... ";
  for (int i = SIZE - 5; i < SIZE; i++)
  {
    cout << sorted[i] << " ";
  }
  cout << "\b]\n";

  double time_taken = double(end - start) / CLOCKS_PER_SEC;
  cout << "Execution time: " << time_taken << " seconds\n";

  return 0;
}