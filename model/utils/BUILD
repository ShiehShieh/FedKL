package(default_visibility = ["//visibility:public"])

py_library(
    name = "utils",
    srcs = ["utils.py"],
)

py_library(
    name = "dict_util",
    srcs = ["dict_util.py"],
)

py_library(
    name = "distributions",
    srcs = ["distributions.py"],
)

py_library(
    name = "prob_type",
    srcs = ["prob_type.py"],
    deps = [
        ":distributions",
        ":utils",
    ],
)

py_library(
    name = "filters",
    srcs = ["filters.py"],
    deps = [
        ":running_stat",
    ],
)

py_library(
    name = "running_stat",
    srcs = ["running_stat.py"],
)

py_library(
    name = "logger_handler",
    srcs = ["logger_handler.py"],
)
