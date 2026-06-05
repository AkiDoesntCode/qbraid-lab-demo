// CUDA-Q x qBraid: GHZ benchmark + observe, run via the qBraid target added
// in https://github.com/NVIDIA/cuda-quantum/pull/4328.
//
// Mirrors `02_qpu_benchmarking.ipynb` in C++: same parameterized GHZ kernel,
// same VQE-style Hamiltonian, same flow you'd use for cross-language parity.
//
// Build and run against the default qBraid QIR state-vector simulator:
// ```
// export QBRAID_API_KEY="qbraid_generated_api_key"
// nvq++ --target qbraid qbraid_example.cpp -o out.x && ./out.x
// ```
//
// To target a different qBraid device, pass its `deviceQrn` via
// `--qbraid-machine`. Browse device IDs at https://account.qbraid.com/devices.
// ```
// nvq++ --target qbraid \
//       --qbraid-machine "qbraid:qbraid:sim:qir-sv" \
//       qbraid_example.cpp -o out.x && ./out.x
// ```
//
// The API key may also be passed inline instead of via the env var:
// ```
// nvq++ --target qbraid \
//       --qbraid-api_key "qbraid_generated_api_key" qbraid_example.cpp
// ```

#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <cudaq/spin_op.h>

#include <cstdio>
#include <fstream>
#include <string>

// ---------------------------------------------------------------------------
// 1. Benchmark kernel: width-N GHZ.
//
// Ideal output distribution is exactly |0...0> and |1...1> with equal
// probability. Any deviation is noise we can score.
// ---------------------------------------------------------------------------
struct ghz {
  void operator()(int n) __qpu__ {
    cudaq::qvector q(n);
    h(q[0]);
    for (int i = 0; i < n - 1; ++i) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

// GHZ fidelity = P(|0...0>) + P(|1...1>). 1.0 on a noiseless device.
static double ghz_fidelity(cudaq::sample_result &counts, int n) {
  const std::string all_zero(n, '0');
  const std::string all_one(n, '1');
  const std::size_t total = counts.get_total_shots();
  if (total == 0)
    return 0.0;
  const std::size_t hits = counts.count(all_zero) + counts.count(all_one);
  return static_cast<double>(hits) / static_cast<double>(total);
}

// ---------------------------------------------------------------------------
// 2. Observe kernel: a 2-qubit VQE-style ansatz with one rotation angle.
// ---------------------------------------------------------------------------
struct ansatz {
  void operator()(double theta) __qpu__ {
    cudaq::qvector q(2);
    x(q[0]);
    ry(theta, q[1]);
    x<cudaq::ctrl>(q[1], q[0]);
  }
};

int main() {
  // ---- A. Synchronous sampling sweep over GHZ widths --------------------
  // Same code path you'd use against a real QPU — only the --qbraid-machine
  // flag at build time changes the target device.
  std::printf("%-8s %-12s %-10s\n", "n", "fidelity", "shots");
  std::printf("---------------------------------\n");
  for (int n : {2, 3, 4, 5}) {
    auto counts = cudaq::sample(/*shots=*/1000, ghz{}, n);
    std::printf("%-8d %-12.3f %-10zu\n", n, ghz_fidelity(counts, n),
                counts.get_total_shots());
  }

  // ---- B. Async submission + future persistence -------------------------
  // QPU runs queue. Submit, write the future to disk, exit, come back later.
  auto future = cudaq::sample_async(ghz{}, 4);
  {
    std::ofstream out("qbraid_future.json");
    out << future;
  }

  // ... time passes, a different process picks the file back up ...
  cudaq::async_result<cudaq::sample_result> readIn;
  {
    std::ifstream in("qbraid_future.json");
    in >> readIn;
  }
  auto async_counts = readIn.get();
  std::printf("\nrehydrated async GHZ(4) counts:\n");
  async_counts.dump();

  // ---- C. Expectation value via observe ---------------------------------
  // Same Hamiltonian and angle as 01_quickstart_integration.ipynb so the
  // C++ and Python outputs are directly comparable.
  using namespace cudaq::spin;
  auto hamiltonian = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     0.21829 * z(0) - 6.125 * z(1);

  auto result = cudaq::observe(/*shots=*/4000, ansatz{}, hamiltonian, 0.59);
  std::printf("\n<H> at theta=0.59: %f\n", result.expectation());

  return 0;
}
