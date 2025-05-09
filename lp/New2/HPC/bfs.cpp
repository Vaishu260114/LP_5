#include <iostream>
#include <queue>
#include <omp.h>
using namespace std;

// Tree node class
class Node {
public:
    int data;
    Node* left;
    Node* right;

    Node(int value) {
        data = value;
        left = right = nullptr;
    }
};

// Binary Tree with BFS functionality
class BinaryTree {
public:
    Node* insertLevelOrder(Node* root, int data);
    void bfsParallel(Node* root);
};

// Insert nodes in level-order fashion
Node* BinaryTree::insertLevelOrder(Node* root, int data) {
    Node* newNode = new Node(data);

    if (!root)
        return newNode;

    queue<Node*> q;
    q.push(root);

    while (!q.empty()) {
        Node* temp = q.front();
        q.pop();

        if (!temp->left) {
            temp->left = newNode;
            return root;
        } else {
            q.push(temp->left);
        }

        if (!temp->right) {
            temp->right = newNode;
            return root;
        } else {
            q.push(temp->right);
        }
    }

    return root;
}

// Perform Breadth-First Search using OpenMP (one level at a time in parallel)
void BinaryTree::bfsParallel(Node* root) {
    if (!root)
        return;

    queue<Node*> q;
    q.push(root);

    cout << "\nBFS Traversal (Parallel per Level):\n";

    while (!q.empty()) {
        int size = q.size();
        vector<Node*> currentLevel;

        // Extract all nodes at current level
        for (int i = 0; i < size; ++i) {
            Node* node = q.front();
            q.pop();
            currentLevel.push_back(node);
        }

        // Process all nodes at this level in parallel
        #pragma omp parallel for
        for (int i = 0; i < currentLevel.size(); ++i) {
            cout << "\t" << currentLevel[i]->data;
        }

        // Enqueue children for next level (serially to avoid race conditions)
        for (Node* node : currentLevel) {
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
    }

    cout << endl;
}

int main() {
    Node* root = nullptr;
    BinaryTree tree;
    int data;
    char ans;

    // Insert nodes interactively
    do {
        cout << "Enter data => ";
        cin >> data;
        root = tree.insertLevelOrder(root, data);
        cout << "Do you want to insert another node? (y/n): ";
        cin >> ans;
    } while (ans == 'y' || ans == 'Y');

    // Perform BFS
    tree.bfsParallel(root);

    return 0;
}
