#include "hapi_kernel_launch.decl.h"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#if defined(BENCHMARK_ENABLE_SAMPLING)
#include <vector>
#endif

#include <converse.h>
#include <hapi.h>

CProxy_Main mainProxy;
CProxy_BenchmarkChare charesProxy;

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

/*readonly*/ int gRangeSize;

constexpr int kDefaultSmCount = 84;
constexpr int kDefaultThreadsPerSm = 1536;


class Main : public CBase_Main {

  private:
    static void timerThunk(void* arg, [[maybe_unused]] double unusedDelayMs) {
      static_cast<Main*>(arg)->thisProxy.stopLaunches();
    }

    CProxy_BenchmarkChare chares_;
    double startTime_ = 0.0;
    int durationSeconds_ = 10;
    bool stopSent_ = false;

  public:
  Main(CkArgMsg* msg) {
    mainProxy = thisProxy;
    Kokkos::initialize();

    const int charesPerThread = parseIntArg(msg, "--chares-per-thread", 1);
    durationSeconds_ = parseIntArg(msg, "--duration-seconds", 10);
    const int smCount = parseIntArg(msg, "--sms", kDefaultSmCount);
    const int threadsPerSm = parseIntArg(msg, "--threads-per-sm", kDefaultThreadsPerSm);
    if (charesPerThread <= 0 || durationSeconds_ <= 0 || smCount <= 0 || threadsPerSm <= 0) {
      CkAbort("All benchmark configuration arguments must be positive.");
    }

    const int threads = std::max(1, CkMyNodeSize());
    const int totalChares = std::max(1, threads * charesPerThread);

    CkPrintf("Launching benchmark with %d threads and %d chares (%d per thread)\n", threads, totalChares,
             charesPerThread);
    gRangeSize = smCount * threadsPerSm;

    CkPrintf("Kernel config: range=%d, duration=%d seconds\n", gRangeSize, durationSeconds_);

    CkArrayOptions opts(totalChares);
    charesProxy = CProxy_BenchmarkChare::ckNew(opts);
    // start() and the timer are deferred until all elements have checked in via allCharesReady().

    delete msg;
  }

  void allCharesReady(CkReductionMsg* msg) {
    delete msg;
    printf("All chares ready. Starting benchmark...\n");
    startTime_ = CkWallTimer();
    CcdCallFnAfter(reinterpret_cast<CcdVoidFn>(Main::timerThunk), this, durationSeconds_ * 1000);
    charesProxy.start();
  }

  void stopLaunches() {
    if (stopSent_) return;
    stopSent_ = true;
    charesProxy.stop();
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

};

class BenchmarkChare : public CBase_BenchmarkChare {
 public:
  BenchmarkChare() {
    createStream(&stream_);
    exec_ = ExecSpace(stream_);
    launchNextCb_ = CkCallback(CkIndex_BenchmarkChare::launchNext(), thisProxy[thisIndex]);
    contribute(CkCallback(CkReductionTarget(Main, allCharesReady), mainProxy));
  }

  ~BenchmarkChare() override { destroyStream(stream_); }

  void start() {
    stopRequested_ = false;
    contributed_ = false;
    launchCount_ = 0;
#if defined(BENCHMARK_ENABLE_SAMPLING)
    gapSamples_.clear();
#endif
#if defined(BENCHMARK_ENABLE_SAMPLING)
    lastSampleTime_ = CkWallTimer();
#endif
    launchNext();
  }

  void stop() { stopRequested_ = true; }

  void launchNext() {
    if (stopRequested_) {
      contributeIfNeeded();
      return;
    }

    //CkPrintf("Chare %d launching kernel %lld\n", thisIndex, launchCount_);
    Kokkos::parallel_for(
        "noop_launch_kernel",
        Kokkos::RangePolicy<ExecSpace>(exec_, 0, gRangeSize),
        // Intentionally empty kernel body to measure launch-rate overhead.
        KOKKOS_LAMBDA(const int) {
        });

    ++launchCount_;
#if defined(BENCHMARK_ENABLE_SAMPLING)
    if (launchCount_ % kSampleInterval == 0) {
      const double now = CkWallTimer();
      gapSamples_.push_back((now - lastSampleTime_) / kSampleInterval);
      lastSampleTime_ = now;
    }
#endif
    hapiAddCallback(stream_, launchNextCb_);
  }

 private:
  void contributeIfNeeded() {
    if (contributed_) return;
    contributed_ = true;
#if defined(BENCHMARK_ENABLE_SAMPLING)
    for (int i = 0; i < static_cast<int>(gapSamples_.size()); ++i) {
      CkPrintf("chare %d sample %d: %.2f us/launch\n",
               thisIndex, i * kSampleInterval, gapSamples_[i] * 1e6);
    }
#endif
    long long local = launchCount_;
    contribute(sizeof(long long), &local, CkReduction::sum_long_long,
               CkCallback(CkReductionTarget(Main, onReduction), mainProxy));
  }

#if defined(BENCHMARK_ENABLE_SAMPLING)
  static constexpr int kSampleInterval = 1000;
  std::vector<double> gapSamples_;
  double lastSampleTime_ = 0.0;
#endif

  CkCallback launchNextCb_;
  NativeStream stream_;
  ExecSpace exec_;
  long long launchCount_ = 0;
  bool stopRequested_ = false;
  bool contributed_ = false;
};

#include "hapi_kernel_launch.def.h"
