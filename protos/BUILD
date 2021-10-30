load(
    "@rules_proto//proto:defs.bzl",
    "proto_library",
)

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
# load("@com_google_googleapis_imports//:imports.bzl", "go_proto_library")
load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "shared_proto",
    srcs = ["shared.proto"],
    deps = [
        "//protos/activity:model_state_proto",
        "//protos/activity:tkpg_proto",
    ],
)

go_proto_library(
    name = "shared_go_proto",
    importpath = "github.com/ShiehShieh/frl/protos/shared_go_proto",
    deps = [
        "//protos/activity:model_state_go_proto",
        "//protos/activity:tkpg_go_proto",
    ],
    protos = [":shared_proto"],
)

py_proto_library(
    name = "shared_py_proto",
    # protos = [":shared_proto"],
    srcs = ["shared.proto"],
    deps = [
        "//protos/activity:model_state_py_proto",
        "//protos/activity:tkpg_py_proto",
    ],
)

proto_library(
    name = "intensive_proto",
    srcs = ["intensive.proto"],
)

go_proto_library(
    name = "intensive_go_proto",
    importpath = "git.llsapp.com/lq/bellman/protos/intensive_go_proto",
    protos = [":intensive_proto"],
)
