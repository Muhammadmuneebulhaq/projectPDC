#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <climits>
#include <algorithm>

using namespace std;

// Structure to store edge
struct Edge {
    int u, v;
    vector<double> weights; // Weights for k objectives
};

// Structure to store SOSP tree node
struct SOSPNode {
    double distance;
    int parent;
    SOSPNode() : distance(1e9), parent(-1) {}
};

// Read MatrixMarket file
vector<Edge> read_mtx(const string& filename, int& num_vertices) {
    vector<Edge> edges;
    ifstream file(filename);
    string line;
    bool header_read = false;
    int num_edges;

    while (getline(file, line)) {
        if (line[0] == '%') continue;
        stringstream ss(line);
        if (!header_read) {
            ss >> num_vertices >> num_vertices >> num_edges;
            header_read = true;
            continue;
        }
        int u, v;
        ss >> u >> v;
        // Convert to 0-based indexing
        edges.push_back({u-1, v-1, {0, 0}});
    }
    file.close();
    return edges;
}

// Assign random weights for k objectives
void assign_weights(vector<Edge>& edges, int k) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(1.0, 10.0);
    for (auto& e : edges) {
        e.weights.resize(k);
        for (int i = 0; i < k; i++) {
            e.weights[i] = dis(gen);
        }
    }
}

// SOSP Update Algorithm (Algorithm 1)
void SOSP_Update(vector<vector<pair<int, double>>>& G, vector<SOSPNode>& T, const vector<Edge>& Ins, int objective) {
    int V = G.size();
    vector<vector<pair<int, double>>> I(V); // Grouped inserted edges
    vector<int> marked(V, 0);
    vector<int> Aff;

    // Step 0: Preprocessing
    for (const auto& e : Ins) {
        I[e.v].push_back({e.u, e.weights[objective]});
    }

    // Step 1: Process Changed Edges
    for (int v = 0; v < V; v++) {
        for (const auto& [u, w] : I[v]) {
            if (T[v].distance > T[u].distance + w) {
                T[v].distance = T[u].distance + w;
                T[v].parent = u;
                Aff.push_back(v);
                marked[v] = 1;
            }
        }
    }

    // Step 2: Propagate the Update
    while (!Aff.empty()) {
        vector<int> N, Aff_new;
        set<int> unique_N;
        for (int v : Aff) {
            for (const auto& [u, w] : G[v]) { // Neighbors of v
                unique_N.insert(u);
            }
        }
        N.assign(unique_N.begin(), unique_N.end());
        for (int v : N) {
            for (const auto& [u, w] : G[v]) { // Predecessors
                if (marked[u] && T[v].distance > T[u].distance + w) {
                    T[v].distance = T[u].distance + w;
                    T[v].parent = u;
                    Aff_new.push_back(v);
                    marked[v] = 1;
                }
            }
        }
        Aff = Aff_new;
    }
}

// Compute initial SOSP using Dijkstra's algorithm
void dijkstra(vector<vector<pair<int, double>>>& G, vector<SOSPNode>& T, int source, int objective) {
    int V = G.size();
    vector<bool> visited(V, false);
    T.assign(V, SOSPNode());
    T[source].distance = 0;

    set<pair<double, int>> pq;
    pq.insert({0, source});

    while (!pq.empty()) {
        int u = pq.begin()->second;
        pq.erase(pq.begin());
        if (visited[u]) continue;
        visited[u] = true;

        for (const auto& [v, w] : G[u]) {
            if (T[v].distance > T[u].distance + w) {
                pq.erase({T[v].distance, v});
                T[v].distance = T[u].distance + w;
                T[v].parent = u;
                pq.insert({T[v].distance, v});
            }
        }
    }
}

// MOSP Update Algorithm (Algorithm 2)
void MOSP_Update(vector<vector<pair<int, double>>>& G, vector<vector<SOSPNode>>& T, const vector<Edge>& Ins, int k) {
    int V = G.size();

    // Step 1: Update SOSP trees
    for (int i = 0; i < k; i++) {
        SOSP_Update(G, T[i], Ins, i);
    }

    // Step 2: Create combined graph
    vector<vector<pair<int, double>>> B(V);
    map<pair<int, int>, int> edge_count;
    for (int i = 0; i < k; i++) {
        for (int v = 0; v < V; v++) {
            if (T[i][v].parent != -1) {
                int u = T[i][v].parent;
                edge_count[{u, v}]++;
            }
        }
    }
    for (const auto& [e, count] : edge_count) {
        int u = e.first, v = e.second;
        B[u].push_back({v, k - count + 1});
    }

    // Step 3: Find SOSP in combined graph
    vector<SOSPNode> T_B;
    dijkstra(B, T_B, 0, 0); // Assuming source is 0

    // Reassign original weights to get MOSP
    vector<pair<int, vector<double>>> MOSP_path;
    int v = V - 1; // Assuming destination is the last vertex
    while (v != -1) {
        int u = T_B[v].parent;
        if (u != -1) {
            for (const auto& e : Ins) {
                if (e.u == u && e.v == v) {
                    MOSP_path.push_back({v, e.weights});
                    break;
                }
            }
        }
        v = u;
    }
    reverse(MOSP_path.begin(), MOSP_path.end());

    // Print MOSP
    cout << "MOSP Path:\n";
    for (const auto& [vertex, weights] : MOSP_path) {
        cout << "Vertex " << vertex + 1 << ": Weights = [";
        for (double w : weights) cout << w << ", ";
        cout << "]\n";
    }
}

int main() {
    string filename = "graph.mtx";
    int num_vertices;
    vector<Edge> edges = read_mtx(filename, num_vertices);
    int k = 2; // Bi-objective
    assign_weights(edges, k);

    // Build adjacency list
    vector<vector<pair<int, double>>> G(num_vertices);
    for (const auto& e : edges) {
        G[e.u].push_back({e.v, e.weights[0]}); // For objective 1
        G[e.v].push_back({e.u, e.weights[0]}); // Symmetric
    }

    // Initialize SOSP trees
    vector<vector<SOSPNode>> T(k);
    for (int i = 0; i < k; i++) {
        dijkstra(G, T[i], 0, i);
    }

    // Generate some inserted edges
    vector<Edge> Ins = {{0, 1, {2.5, 3.0}}, {1, 2, {1.5, 2.0}}};
    assign_weights(Ins, k);

    // Update MOSP
    MOSP_Update(G, T, Ins, k);

    return 0;
}
