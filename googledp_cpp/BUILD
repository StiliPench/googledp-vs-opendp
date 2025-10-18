cc_binary(
    name = "dp_comparison",
    srcs = ["main.cc"],
    deps = [
        "@differential_privacy_cc//algorithms:bounded-sum",
        "@differential_privacy_cc//algorithms:count",
        "@com_google_absl//absl/status:statusor",
    ],
    copts = ["-std=c++17"]
)
