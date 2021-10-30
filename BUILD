package(default_visibility = ["//visibility:public"])

load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_proto//proto:defs.bzl", "proto_library")

py_binary(
    name = "main",
    srcs = [
        "main.py",
    ],
    deps = [
        "//client:client",
        "//env:airraidramv0",
        "//env:antv2",
        "//env:cartpolev0",
        "//env:fetchpickandplacev1",
        "//env:halfcheetahv2",
        "//env:hopperv2",
        "//env:humanoidv2",
        "//env:invertedpendulumv2",
        "//env:walker2dv2",
        "//model/fl:fedavg",
        "//model/fl:fedprox",
        "//model/fl:fedtrpo",
        "//model/rl/agent:agent",
        "//model/rl/agent:critic",
        "//model/rl/agent:reinforce",
        "//model/rl/agent:trpo",
        "//model/optimizer:pgd",
    ],
)
