package(default_visibility = ["//visibility:public"])

py_library(
    name = "fedbase",
    srcs = [
        "fedbase.py",
    ],
    deps = [
        "//model/rl/agent:vec_agent",
        "//model/utils:vectorization",
    ],
)

py_library(
    name = "fedavg",
    srcs = [
        "fedavg.py",
    ],
    deps = [
        ":fedbase",
        "//model/rl/agent:vec_agent",
        "//model/utils:vectorization",
    ],
)

py_library(
    name = "fedprox",
    srcs = [
        "fedprox.py",
    ],
    deps = [
        ":fedbase",
        "//model/rl/agent:vec_agent",
        "//model/utils:vectorization",
    ],
)

py_library(
    name = "fedtrpo",
    srcs = [
        "fedtrpo.py",
    ],
    deps = [
        ":fedbase",
        "//model/rl/agent:vec_agent",
        "//model/utils:vectorization",
    ],
)
