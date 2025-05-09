#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>
using namespace std;

void parallelDFS(const vector<vector<int>>& graph, vector<bool>& visited, int start) {
    stack<int> s;
    s.push(start);
    visited[start] = true;

    while (!s.empty()) {
        int curr = s.top();
        s.pop();
        cout << curr << " ";

        #pragma omp parallel for
        for (int neighbor : graph[curr]) {
            #pragma omp critical
            {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    s.push(neighbor);
                }
            }
        }
    }
}

int main() {
    int n, m, start;
    cout << "Enter number of nodes, edges, and start node: ";
    cin >> n >> m >> start;

    vector<vector<int>> graph(n);
    vector<bool> visited(n, false);

    cout << "Enter edges (u v):\n";
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u); // For undirected graph
    }

    cout << "DFS traversal: ";
    parallelDFS(graph, visited, start);

    return 0;
}