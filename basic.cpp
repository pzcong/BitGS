#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>
using namespace std;
#include "BitGS.h"
int main() {
    // Synthetic usage example, not a paper benchmark.
    BitGS sketch(1000000, 3, 16, 4,
        {{8,4,16}, {80,4,2}, {100,8,2}, {40,8,2}, {20,8,2}},
        4096, 4096, 4096);
    sketch.insert(1, 2, 4);
    sketch.insert(1, 3, 2);
    sketch.insert(1, 2, 1);
    const auto estimate = sketch.edge_query(1, 2);
    cout << "edge(1,2): " << estimate << "\n";
    return estimate == 5 ? 0 : 1;
}
