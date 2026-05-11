#include "hapi_kernel_launch.decl.h"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

#include <converse.h>
#include <hapi.h>

#if defined(BENCHMARK_USE_CUDA)
#include <cuda_runtime.h>
using NativeStream = cudaStream_t;
static void createStream(NativeStream* stream) {
  const auto status = cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
  if (status != cudaSuccess) {
    CkAbort(cudaGetErrorString(status));
  }
}
static void destroyStream(NativeStream stream) {
  const auto status = cudaStreamDestroy(stream);
  if (status != cudaSuccess) {
    CkAbort(cudaGetErrorString(status));
  }
}
using ExecSpace = Kokkos::Cuda;
#elif defined(BENCHMARK_USE_HIP)
#include <hip/hip_runtime.h>
using NativeStream = hipStream_t;
static void createStream(NativeStream* stream) {
  const auto status = hipStreamCreateWithFlags(stream, hipStreamNonBlocking);
  if (status != hipSuccess) {
    CkAbort(hipGetErrorString(status));
  }
}
static void destroyStream(NativeStream stream) {
  const auto status = hipStreamDestroy(stream);
  if (status != hipSuccess) {
    CkAbort(hipGetErrorString(status));
  }
}
using ExecSpace = Kokkos::HIP;
#else
#error "Define BENCHMARK_USE_CUDA or BENCHMARK_USE_HIP"
#endif

static int parseIntArg(CkArgMsg* msg, const char* name, int defaultValue) {
  const std::string prefix = std::string(name) + "=";
  for (int i = 1; i < msg->argc; ++i) {
    if (std::strncmp(msg->argv[i], prefix.c_str(), prefix.size()) == 0) {
      return std::atoi(msg->argv[i] + prefix.size());
    }
  }
  return defaultValue;
}

/*readonly*/ int gLeagueSize;
/*readonly*/ int gTeamSize;

constexpr int kDefaultSmCount = 84;
constexpr int kDefaultThreadsPerSm = 1536;
constexpr int kDefaultBlocksPerSm = 16;

class Main : public CBase_Main {
 public:
  Main(CkArgMsg* msg) {
    Kokkos::initialize();

    const int charesPerThread = parseIntArg(msg, "--chares-per-thread", 1);
    durationSeconds_ = parseIntArg(msg, "--duration-seconds", 10);
    const int smCount = parseIntArg(msg, "--sms", kDefaultSmCount);
    const int threadsPerSm = parseIntArg(msg, "--threads-per-sm", kDefaultThreadsPerSm);
    const int blocksPerSm = parseIntArg(msg, "--blocks-per-sm", kDefaultBlocksPerSm);
    if (charesPerThread <= 0 || durationSeconds_ <= 0 || smCount <= 0 || threadsPerSm <= 0 || blocksPerSm <= 0) {
      CkAbort("All benchmark configuration arguments must be positive.");
    }

    const int threads = std::max(1, CkMyNodeSize());
    const int totalChares = std::max(1, threads * charesPerThread);
    const int totalBlocks = std::max(1, smCount * blocksPerSm);
    // Rounded-up division to derive threads-per-block from per-SM occupancy targets.
    const int teamSize = std::max(1, (threadsPerSm + blocksPerSm - 1) / blocksPerSm);

    CkPrintf("Launching benchmark with %d threads and %d chares (%d per thread)\n", threads, totalChares,
             charesPerThread);
    gLeagueSize = totalBlocks;
    gTeamSize = teamSize;

    CkPrintf("Kernel config: league=%d, team=%d, duration=%d seconds\n", gLeagueSize, gTeamSize, durationSeconds_);

    CkArrayOptions opts(totalChares);
    chares_ = CProxy_BenchmarkChare::ckNew(opts);
    chares_.start();

    startTime_ = CkWallTimer();
    CcdCallFnAfter(reinterpret_cast<CcdVoidFn>(Main::timerThunk), this, durationSeconds_ * 1000);

    delete msg;
  }

  void stopLaunches() {
    if (stopSent_) return;
    stopSent_ = true;
    chares_.stop();
  }

  void onReduction(CkReductionMsg* msg) {
    const auto* totalLaunches = static_cast<const long long*>(msg->getData());
    const double elapsed = CkWallTimer() - startTime_;
    const double launchesPerSecond = (*totalLaunches) / elapsed;
    CkPrintf("Total kernel launches: %lld\n", *totalLaunches);
    CkPrintf("Elapsed seconds: %.6f\n", elapsed);
    CkPrintf("Launches per second: %.3f\n", launchesPerSecond);
    delete msg;
    Kokkos::finalize();
    CkExit();
  }

 private:
  static void timerThunk(void* arg, double unusedDelayMs) {
    (void)unusedDelayMs;
    static_cast<Main*>(arg)->thisProxy.stopLaunches();
  }

  CProxy_BenchmarkChare chares_;
  double startTime_ = 0.0;
  int durationSeconds_ = 10;
  bool stopSent_ = false;
};

class BenchmarkChare : public CBase_BenchmarkChare {
 public:
  BenchmarkChare() : mainProxy_(CProxy_Main(0)) { createStream(&stream_); }

  ~BenchmarkChare() override { destroyStream(stream_); }

  void start() {
    stopRequested_ = false;
    contributed_ = false;
    launchCount_ = 0;
    launchNext();
  }

  void stop() { stopRequested_ = true; }

  void launchNext() {
    if (stopRequested_) {
      contributeIfNeeded();
      return;
    }

    ExecSpace exec(stream_);
    Kokkos::parallel_for(
        "noop_launch_kernel",
        Kokkos::TeamPolicy<ExecSpace>(exec, gLeagueSize, gTeamSize),
        // Intentionally empty kernel body to measure launch-rate overhead.
        KOKKOS_LAMBDA([[maybe_unused]] const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
          (void)teamMember;
        });

    ++launchCount_;
    hapiAddCallback(stream_, CkCallback(CkIndex_BenchmarkChare::launchNext(), thisProxy[thisIndex]));
  }

 private:
  void contributeIfNeeded() {
    if (contributed_) return;
    contributed_ = true;
    long long local = launchCount_;
    contribute(sizeof(long long), &local, CkReduction::sum_long_long,
               CkCallback(CkReductionTarget(Main, onReduction), mainProxy_));
  }

  CProxy_Main mainProxy_;
  NativeStream stream_;
  long long launchCount_ = 0;
  bool stopRequested_ = false;
  bool contributed_ = false;
};

#include "hapi_kernel_launch.def.h"
