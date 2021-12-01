package(default_visibility = ["//visibility:public"])

load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_proto//proto:defs.bzl", "proto_library")

py_binary(
    name = "federated",
    srcs = [
        "federated.py",
    ],
    deps = [
        "//client:client",
        "//environment:airraidramv0",
        "//environment:antv2",
        "//environment:cartpolev0",
        "//environment:fetchpickandplacev1",
        "//environment:halfcheetahv2",
        "//environment:hopperv2",
        "//environment:humanoidv2",
        "//environment:invertedpendulumv2",
        "//environment:reacherv2",
        "//environment:walker2dv2",
        "//model/rl/agent:agent",
        "//model/rl/agent:critic",
        "//model/rl/agent:reinforce",
        "//model/rl/agent:trpo",
        "//model/optimizer:pgd",
    ],
)

py_binary(
    name = "main",
    srcs = [
        "main.py",
    ],
    deps = [
        "//client:client",
        "//environment:airraidramv0",
        "//environment:antv2",
        "//environment:cartpolev0",
        "//environment:fetchpickandplacev1",
        "//environment:halfcheetahv2",
        "//environment:hopperv2",
        "//environment:humanoidv2",
        "//environment:invertedpendulumv2",
        "//environment:reacherv2",
        "//environment:walker2dv2",
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

py_binary(
    name = "inspect_heterogeneity",
    srcs = [
        "inspect_heterogeneity.py",
    ],
    deps = [
        "//client:client",
        "//environment:airraidramv0",
        "//environment:antv2",
        "//environment:cartpolev0",
        "//environment:fetchpickandplacev1",
        "//environment:halfcheetahv2",
        "//environment:hopperv2",
        "//environment:humanoidv2",
        "//environment:invertedpendulumv2",
        "//environment:reacherv2",
        "//environment:walker2dv2",
        "//model/fl:fedavg",
        "//model/fl:fedprox",
        "//model/fl:fedtrpo",
        "//model/rl/agent:agent",
        "//model/rl/agent:critic",
        "//model/rl/agent:reinforce",
        "//model/rl/agent:trpo",
        "//model/rl/comp:state_visitation_frequency",
        "//model/optimizer:pgd",
    ],
)
