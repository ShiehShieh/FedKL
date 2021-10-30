package(default_visibility = ["//visibility:public"])

py_binary(
    name = "train",
    srcs = [
        "train.py",
        "//protos/activity:model_constants_py_proto",
    ],
    deps = [
        "//model/inputs:inputs",
        "//model/inputs:model_constants",
        "//model/rl/agent:actor",
        "//model/rl/agent:agent",
        "//model/rl/agent:ops",
        "//model/utils:utils",
    ],
)

py_binary(
    name = "train_gym",
    srcs = [
        "train_gym.py",
    ],
    deps = [
        "//model/rl/agent:actor",
        "//model/rl/agent:agent",
        "//model/rl/agent:ops",
    ],
)
