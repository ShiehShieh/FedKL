load(
    "@rules_proto//proto:defs.bzl",
    "proto_library",
)

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "sample_proto",
    srcs = ["sample.proto"],
    deps = [
        "//protos:shared_proto",
        "//protos/activity:tkpg_proto",
    ],
)

go_proto_library(
    name = "sample_go_proto",
    importpath = "github.com/ShiehShieh/frl/protos/activity/sample_go_proto",
    deps = [
        ":tkpg_go_proto",
        "//protos:shared_go_proto",
    ],
    protos = [":sample_proto"],
)

proto_library(
    name = "tkpg_proto",
    srcs = ["tkpg.proto"],
)

go_proto_library(
    name = "tkpg_go_proto",
    importpath = "github.com/ShiehShieh/frl/protos/activity/tkpg_go_proto",
    protos = [":tkpg_proto"],
)

py_proto_library(
    name = "tkpg_py_proto",
    # protos = [":tkpg_proto"],
    srcs = ["tkpg.proto"],
)

proto_library(
    name = "model_state_proto",
    srcs = ["model_state.proto"],
    deps = ["//protos/activity:tkpg_proto"],
)

go_proto_library(
    name = "model_state_go_proto",
    importpath = "github.com/ShiehShieh/frl/protos/activity/model_state_go_proto",
    deps = [
        ":tkpg_go_proto",
    ],
    protos = [":model_state_proto"],
)

py_proto_library(
    name = "model_state_py_proto",
    # protos = [":model_state_proto"],
    deps = [
        ":tkpg_py_proto",
    ],
    srcs = ["model_state.proto"],
)

proto_library(
    name = "model_constants_proto",
    srcs = ["model_constants.proto"],
)

go_proto_library(
    name = "model_constants_go_proto",
    importpath = "github.com/ShiehShieh/frl/protos/activity/model_constants_go_proto",
    protos = [":model_constants_proto"],
)

py_proto_library(
    name = "model_constants_py_proto",
    # protos = [":model_constants_proto"],
    srcs = ["model_constants.proto"],
)
