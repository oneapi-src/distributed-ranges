# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import re
from argparse import ArgumentParser
from glob import iglob
from os.path import abspath
from sys import exit

args = None
errors = 0

logger = logging.getLogger("oneapi-style-check")

ch = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False

exclude_rule = re.compile("dr-style ignore")

pragma_once_rule = re.compile("#pragma once")

all_rules = [
    (
        r"lib::",
        "use dr::",
    ),
    (
        r"namespace lib {",
        "use namespace dr {",
    ),
    (
        r"// namespace lib",
        "use // namespace dr",
    ),
    (
        r"namespace mhp {",
        "use namespace dr::mhp {",
    ),
    (
        r"// namespace mhp",
        "use // namespace dr::mhp",
    ),
    (
        r"namespace shp {",
        "use namespace dr::shp {",
    ),
    (
        r"// namespace shp",
        "use // namespace dr::shp",
    ),
    (
        r"[( ]size_t ",
        "use std::size_t",
    ),
    (
        r" string ",
        "use std::string",
    ),
    (
        r"using namespace sycl|using namespace cl::sycl",
        "eliminate using",
    ),
    (
        r"malloc_shared\(|malloc_device\(|malloc_host\(",
        "use templated malloc",
    ),
    (r"#include <CL/sycl.hpp>", "use #include <sycl/sycl.hpp>"),
    (r"\(.*?\)malloc", "use static_cast or new"),
    (r"namespace sycl = ", "do not define sycl namespace"),
    (r"[< (]cl::sycl", "delete cl::"),
    (r"parallel_for<class", "use unnamed lambda"),
    (r"sycl::write_only\)", "use sycl::write_only, sycl::no_init)"),
    (
        r"\.get_access<sycl::access::mode::read",
        "use constructor instead of get_access",
    ),
    (
        r"\.get_access<sycl::access::mode::write",
        "use constructor instead of get_access",
    ),
    (
        r"\.get_host_access",
        "use constructor instead of get_host_access",
    ),
    (
        r"std::input_iterator( |<)",
        "use std_dr::input_iterator",
    ),
    (
        r"std::output_iterator( |<)",
        "use std_dr::output_iterator",
    ),
    (
        r"std::bidirectional_iterator( |<)",
        "use std_dr::bidirectional_iterator",
    ),
    (
        r"std::contiguous_iterator( |<)",
        "use std_dr::contiguous_iterator",
    ),
    (
        r"std::random_access_iterator( |<)",
        "use std_dr::random_access_iterator",
    ),
    (
        r"std_dr::",
        "use std_dr::input_iterator",
    ),
]

all_rules_compiled = []

include_rules = [
    (
        r'#include "',
        "use #include <>",
    ),
    (
        r"namespace {",
        "use namespace __detail {",
    ),
    (
        r"namespace internal {",
        "use namespace __detail {",
    ),
    (
        r"(?<!comm)\.size\(\)",
        "use rng::size()",
    ),
    (
        r"\.empty\(\)",
        "use rng::empty()",
    ),
    (
        r"std::begin\(",
        "use rng::begin()",
    ),
    (
        r"std::end\(",
        "use rng::end()",
    ),
    (
        r"std::distance\(",
        "use rng::distance()",
    ),
    (
        r"std::size\(",
        "use rng::size()",
    ),
]
include_rules_compiled = []


def parse_args():
    parser = ArgumentParser(description="Check for oneapi coding style issues")
    parser.add_argument(
        "directory",
        default=".",
        nargs="+",
        help="root directory for recursive search",
    )
    parser.add_argument(
        "--Werror", action="store_true", help="treat warnings as errors"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="debugging output"
    )
    parser.add_argument(
        "--include", action="store_true", help="use include directory rules"
    )
    return parser.parse_args()


def warning(warning, file, line, message):
    global errors
    if args.Werror:
        errors += 1
    logger.warning(f"{file}:{line}: warning: {warning}: {message}")


def check_file(path):
    logging.info(f"Checking {path}")
    rules = include_rules_compiled if args.include else all_rules_compiled
    lineno = 1
    with open(path) as fin:
        for line in fin.readlines():
            for rule in rules:
                if rule[0].search(line) and not exclude_rule.search(line):
                    warning(rule[1], path, lineno, line.rstrip())
            lineno += 1


def check_header_file(path):
    logging.info(f"Checking header {path}")
    with open(path) as fin:
        found_pragma_once = False
        for line in fin.readlines():
            if pragma_once_rule.search(line):
                found_pragma_once = True
                break
        if not found_pragma_once:
            warning("add #pragma once", path, "", "")


def check_files():
    for dir in args.directory:
        for file in iglob(f"{dir}/**/*.[ch]pp", recursive=True):
            check_file(abspath(file))
        for file in iglob(f"{dir}/**/*.hpp", recursive=True):
            check_header_file(abspath(file))


def compile_checks():
    global all_rules_compiled, include_rules_compiled
    all_rules_compiled = [
        (re.compile(check[0]), check[1]) for check in all_rules
    ]
    include_rules_compiled = [
        (re.compile(check[0]), check[1]) for check in include_rules
    ]


def main():
    global args
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    compile_checks()
    check_files()
    print(f"Errors: {errors}")
    exit(errors > 0)


if __name__ == "__main__":
    main()
