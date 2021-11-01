package(default_visibility = ["//visibility:public"])

py_library(
    name = "vec_env",
    srcs = ["__init__.py"],
    deps = [
        ":base_vec_env",
        ":dummy_vec_env",
        ":subproc_vec_env",
        ":vec_frame_stack",
        ":vec_normalize",
        ":vec_video_recorder",
        ":vec_check_nan",
    ],
)

py_library(
    name = "logger",
    srcs = ["logger.py"],
)

py_library(
    name = "tile_images",
    srcs = ["tile_images.py"],
)

py_library(
    name = "util",
    srcs = ["util.py"],
)

py_library(
    name = "running_mean_std",
    srcs = ["running_mean_std.py"],
)

py_library(
    name = "base_vec_env",
    srcs = ["base_vec_env.py"],
    deps = [
        ":logger",
        ":tile_images",
    ],
)

py_library(
    name = "dummy_vec_env",
    srcs = ["dummy_vec_env.py"],
    deps = [
        ":base_vec_env",
        ":util",
    ],
)

py_library(
    name = "subproc_vec_env",
    srcs = ["subproc_vec_env.py"],
    deps = [
        ":base_vec_env",
    ],
)

py_library(
    name = "vec_check_nan",
    srcs = ["vec_check_nan.py"],
    deps = [
        ":base_vec_env",
    ],
)

py_library(
    name = "vec_frame_stack",
    srcs = ["vec_frame_stack.py"],
    deps = [
        ":base_vec_env",
    ],
)

py_library(
    name = "vec_normalize",
    srcs = ["vec_normalize.py"],
    deps = [
        ":base_vec_env",
        ":running_mean_std",
    ],
)

py_library(
    name = "vec_video_recorder",
    srcs = ["vec_video_recorder.py"],
    deps = [
        ":base_vec_env",
        ":dummy_vec_env",
        ":logger",
        ":running_mean_std",
        ":subproc_vec_env",
        ":vec_frame_stack",
        ":vec_normalize",
    ],
)
