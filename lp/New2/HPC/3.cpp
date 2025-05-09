#include <iostream>
#include <vector>
#include <omp.h>
#include <climits>

using namespace std;

vector<int> get_user_input()
{
  vector<int> arr;
  int size, value;

  cout << "Enter number of elements: ";
  cin >> size;

  if (size <= 0)
  {
    return arr;
  }

  cout << "Enter " << size << " integers:\n";
  for (int i = 0; i < size; i++)
  {
    cin >> value;
    arr.push_back(value);
  }

  return arr;
}

void print_array(const vector<int> &arr)
{
  cout << "\nInput array: [";
  for (int i = 0; i < arr.size(); i++)
  {
    cout << arr[i];
    if (i != arr.size() - 1)
      cout << ", ";
  }
  cout << "]";
}

int find_min(const vector<int> &arr)
{
  int min_val = INT_MAX;
#pragma omp parallel for reduction(min : min_val)
  for (int i = 0; i < arr.size(); i++)
  {
    if (arr[i] < min_val)
    {
      min_val = arr[i];
    }
  }
  return min_val;
}

int find_max(const vector<int> &arr)
{
  int max_val = INT_MIN;
#pragma omp parallel for reduction(max : max_val)
  for (int i = 0; i < arr.size(); i++)
  {
    if (arr[i] > max_val)
    {
      max_val = arr[i];
    }
  }
  return max_val;
}

int calculate_sum(const vector<int> &arr)
{
  int sum = 0;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < arr.size(); i++)
  {
    sum += arr[i];
  }
  return sum;
}

double calculate_average(const vector<int> &arr)
{
  int sum = calculate_sum(arr);
  return static_cast<double>(sum) / arr.size();
}

int main()
{
  vector<int> arr = get_user_input();

  if (arr.empty())
  {
    cout << "No elements entered. Exiting...\n";
    return 1;
  }

  print_array(arr);

  cout << "\nResults:\n";
  cout << "Minimum value: " << find_min(arr) << endl;
  cout << "Maximum value: " << find_max(arr) << endl;
  cout << "Sum: " << calculate_sum(arr) << endl;
  cout << "Average: " << calculate_average(arr) << endl;

  return 0;
}