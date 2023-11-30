import math


def get(val: float) -> str:
    return {
        0: "uniform",
        1: "normal",
        2: "lognormal",
    }[int(val)]


def transform(
    dist: str,
    mean: float,
    std: float,
) -> dict:
    if dist == "uniform":
        lb = mean - (math.sqrt(12) * std) / 2
        ub = mean + (math.sqrt(12) * std) / 2
        return {
            "a": lb,
            "b": ub,
        }

    elif dist == "normal":
        return {
            "mu": mean,
            "sd": std,
        }

    elif dist == "lognormal":
        sd = math.sqrt(math.log((std / mean) ** 2 + 1))
        mu = math.log(mean) - 0.5 * math.log((std / mean) ** 2 + 1)

        return {
            "mu": mu,
            "sd": sd,
        }

    else:
        raise NotImplementedError(f"Unknown distribution: {dist}")
