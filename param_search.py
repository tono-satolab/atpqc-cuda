"""
Searches optimized parameters from benchmark output.

param_search.py

Copyright (c) 2021 Tatsuki Ono

This software is released under the MIT License.
https://opensource.org/licenses/mit-license.php
"""

import os.path
import sys

import numpy as np


def read_perf_file(fname):
    csv_dat = np.loadtxt(
        fname,
        dtype={
            "names": (
                "ninputs",
                "genmat_nwarps",
                "genvec_nwarps",
                "genpoly_nwarps",
                "sha3_nwarps",
                "keypair_latency",
                "keypair_throughput",
                "enc_latency",
                "enc_throughput",
                "dec_latency",
                "dec_throughput",
            ),
            "formats": ("I", "I", "I", "I", "I", "g", "g", "g", "g", "g", "g"),
        },
        delimiter=",",
    )

    params = np.empty(
        len(csv_dat),
        dtype={
            "names": (
                "ninputs",
                "genmat_nwarps",
                "genvec_nwarps",
                "genpoly_nwarps",
                "sha3_nwarps",
            ),
            "formats": ("I", "I", "I", "I", "I"),
        },
    )
    params["ninputs"] = csv_dat["ninputs"]
    params["genmat_nwarps"] = csv_dat["genmat_nwarps"]
    params["genvec_nwarps"] = csv_dat["genvec_nwarps"]
    params["genpoly_nwarps"] = csv_dat["genpoly_nwarps"]
    params["sha3_nwarps"] = csv_dat["sha3_nwarps"]

    keypair_result = np.empty(
        len(csv_dat),
        dtype={"names": ("latency", "throughput"), "formats": ("g", "g")},
    )
    keypair_result["latency"] = csv_dat["keypair_latency"]
    keypair_result["throughput"] = csv_dat["keypair_throughput"]

    enc_result = np.empty(
        len(csv_dat),
        dtype={"names": ("latency", "throughput"), "formats": ("g", "g")},
    )
    enc_result["latency"] = csv_dat["enc_latency"]
    enc_result["throughput"] = csv_dat["enc_throughput"]

    dec_result = np.empty(
        len(csv_dat),
        dtype={"names": ("latency", "throughput"), "formats": ("g", "g")},
    )
    dec_result["latency"] = csv_dat["dec_latency"]
    dec_result["throughput"] = csv_dat["dec_throughput"]

    return (params, keypair_result, enc_result, dec_result)


def get_optimal(params, results, func):
    ev = func(results["latency"], results["throughput"])
    i_max = np.argmax(ev)
    return (params[i_max], results[i_max], ev[i_max])


def threshold(x, req, cap):
    ev = x.copy()
    ev[x < req] = 0
    ev[x > cap] = cap
    return ev


def eval_l_per_t(latency, throughput):
    return throughput / latency


def eval_latency_cap(latency, throughput):
    delta = 30e-3 / 3
    ev = throughput * threshold(1 / latency, 1 / delta, 1 / delta)
    return ev


def eval_throughput_req(latency, throughput):
    req = 2 ** 15
    ev = 1 / latency * threshold(throughput, req, req)
    return ev


def eval_latency(latency, throughput):
    return 1 / latency


def eval_throughput(latency, throughput):
    return throughput


def print_opt_user(params, results, ev):
    print(
        "ninputs: {ninputs}, "
        "nwarps(genmat, genvec, genpoly, sha3): "
        "({genmat_nwarps}, {genvec_nwarps}, "
        "{genpoly_nwarps}, {sha3_nwarps})".format_map(params)
    )
    print(
        "latency: {0:e}, throughput: {1:e}, evaluation: {2:e}".format(
            results["latency"], results["throughput"], ev
        )
    )


def print_opt_params(opt):
    print(
        "{ninputs} "
        "{genmat_nwarps} "
        "{genvec_nwarps} "
        "{genpoly_nwarps} "
        "{sha3_nwarps}".format_map(opt[0])
    )


def exec_opt_print(params, keypair_result, enc_result, dec_result, fopt):
    keypair_opt = get_optimal(params, keypair_result, fopt)
    enc_opt = get_optimal(params, enc_result, fopt)
    dec_opt = get_optimal(params, dec_result, fopt)

    print("Keypair:")
    print_opt_user(*keypair_opt)
    print()
    print("Encaps (Bob):")
    print_opt_user(*enc_opt)
    print()
    print("Decaps:")
    print_opt_user(*dec_opt)
    print()
    print(
        "All: l={0:e}, c={1:e}".format(
            keypair_opt[1]["latency"] + enc_opt[1]["latency"] + dec_opt[1]["latency"],
            np.amin(
                (
                    keypair_opt[1]["throughput"],
                    enc_opt[1]["throughput"],
                    dec_opt[1]["throughput"],
                )
            ),
        )
    )
    print()
    print(
        "Alice: l={0:e}, c={1:e}".format(
            keypair_opt[1]["latency"] + dec_opt[1]["latency"],
            np.amin(
                (
                    keypair_opt[1]["throughput"],
                    dec_opt[1]["throughput"],
                )
            ),
        )
    )
    print()
    # print_opt_params(keypair_opt)
    # print_opt_params(enc_opt)
    # print_opt_params(dec_opt)
    # print()


if __name__ == "__main__":
    result_dir = "./target/bench"

    print("Select variant:")
    print("\tKyber512  = 0")
    print("\tKyber768  = 1")
    print("\tKyber1024 = 2")
    print("\t    (0-2) =")
    variant = int(input())
    if variant == 0:
        print("Selected variant: Kyber512")
        fname = "bench_kyber512_result.txt"
    elif variant == 1:
        print("Selected variant: Kyber768")
        fname = "bench_kyber768_result.txt"
    elif variant == 2:
        print("Selected variant: Kyber1024")
        fname = "bench_kyber1024_result.txt"
    else:
        print("Please type 0-2.")
        sys.exit(0)
    print("")

    fpath = os.path.join(result_dir, fname)
    (params, keypair_result, enc_result, dec_result) = read_perf_file(fpath)

    print("Custom evaluation mode (type '1' to enable):")

    def is_custom_mode(s):
        try:
            mode = int(s)
        except ValueError:
            return False
        else:
            if mode == 1:
                return True
            else:
                return False

    if is_custom_mode(input()):
        print("Selected mode: Custom mode")
        print("")
        print("Type delta_req:")
        delta_req = float(input())
        print("Type delta_cap:")
        delta_cap = float(input())
        print("Type lambda_req:")
        lambda_req = float(input())
        print("Type lambda_cap:")
        lambda_cap = float(input())
        print("")
        print(
            "(delta_req, delta_cap, lambda_req, lambda_cap) = ({}, {}, {}, {})".format(
                delta_req, delta_cap, lambda_req, lambda_cap
            )
        )
        print("")

        exec_opt_print(
            params,
            keypair_result,
            enc_result,
            dec_result,
            lambda l, c: threshold(1 / l, 1 / delta_req, 1 / delta_cap)
            * threshold(c, lambda_req, lambda_cap),
        )

    else:
        print("Selected mode: Default mode")
        print("")
        print("--- c/l ---")
        exec_opt_print(params, keypair_result, enc_result, dec_result, eval_l_per_t)
        print("--- maximize c under l < 10ms * 3 ---")
        exec_opt_print(params, keypair_result, enc_result, dec_result, eval_latency_cap)
        print("--- minimize l under c > 32K ---")
        exec_opt_print(
            params, keypair_result, enc_result, dec_result, eval_throughput_req
        )
        print("--- minimize l ---")
        exec_opt_print(params, keypair_result, enc_result, dec_result, eval_latency)
        print("--- maximize c ---")
        exec_opt_print(params, keypair_result, enc_result, dec_result, eval_throughput)
