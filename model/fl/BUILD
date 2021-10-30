package(default_visibility = ["//visibility:public"])

py_library(
    name = "fedbase",
    srcs = [
        "fedbase.py",
    ],
)

py_library(
    name = "fedavg",
    srcs = [
        "fedavg.py",
    ],
    deps = [
        ":fedbase",
    ],
)

py_library(
    name = "fedprox",
    srcs = [
        "fedprox.py",
    ],
    deps = [
        ":fedbase",
    ],
)

py_library(
    name = "fedtrpo",
    srcs = [
        "fedtrpo.py",
    ],
    deps = [
        ":fedbase",
    ],
)
