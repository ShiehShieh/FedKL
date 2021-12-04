package(default_visibility = ["//visibility:public"])

py_library(
    name = "ops",
    srcs = ["ops.py"],
)

py_library(
    name = "actor",
    srcs = ["actor.py"],
    deps = [
        "//model/rl/comp:cfn_cell",
        "//model/utils:utils",
    ],
)

py_library(
    name = "agent",
    srcs = ["agent.py"],
    deps = [
        ":actor",
        ":ops",
    ],
)

py_library(
    name = "pg",
    srcs = ["pg.py"],
)

py_library(
    name = "reinforce",
    srcs = ["reinforce.py"],
    deps = [
        ":pg",
        "//model/utils:distributions",
        "//model/utils:utils",
    ],
)

py_library(
    name = "trpo",
    srcs = ["trpo.py"],
    deps = [
        ":pg",
        "//model/utils:filters",
        "//model/utils:distributions",
        "//model/utils:prob_type",
        "//model/utils:utils",
    ],
)

py_library(
    name = "critic",
    srcs = ["critic.py"],
    deps = [
    ],
)

py_library(
    name = "vec_agent",
    srcs = ["vec_agent.py"],
    deps = [
    ],
)
