#!/usr/bin/python3

import argparse

def str2bool(value):
    """A function to convert string to bool value."""
    if value.lower() in {'yes', 'true', 't', 'y', '1'}:
        return True
    if value.lower() in {'no', 'false', 'f', 'n', '0'}:
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    """Parse arguments.

    Returns:
    A tuple of:
        - `tree_inference_args`: all arguments for tree-based inferencing
    """

    # tree_inference configurations
    tree_inference_parser = argparse.ArgumentParser(
        description='Common configurations.', add_help=False)

    tree_inference_parser.add_argument(
        '--enable_regression',
        default='true',
        type=str2bool,
        help='Enables regression with numerical features. Default: true')
    tree_inference_parser.add_argument(
        '--enable_binomial',
        default='false',
        type=str2bool,
        help='Enables binomial with numerical features. Default: false')
    tree_inference_parser.add_argument(
        '--enable_multinomial',
        default='false',
        type=str2bool,
        help='Enables multinomial with numerical features. Default: false')
    tree_inference_parser.add_argument(
        '--enable_SW_inference',
        default='true',
        type=str2bool,
        help='Enables CPU-based inference. Default: true')
    tree_inference_parser.add_argument(
        '--enable_FPGA_inference',
        default='true',
        type=str2bool,
        help='Enables FPGA-based inference. Default: true')
    tree_inference_parser.add_argument(
        '--max_depth',
        type=int,
        default=8,
        help='Maximum number of tree levels for training. Default: 8')
    tree_inference_parser.add_argument(
        '--number_of_trees',
        type=int,
        default=100,
        help='Number of trees per forest for training. Default: 100')
    tree_inference_parser.add_argument(
        '--num_test_samples',
        type=int,
        default=100,
        help='Number of samples per forest for inference. Default: 100')
    tree_inference_parser.add_argument(
        '--n_loops',
        type=int,
        default=1000,
        help='Number of iterations over the same data for a better average execution time. Default: 1000')
    tree_inference_parser.add_argument(
        '--data_fpath',
        type=str,
        default='/home/centos/xelera-demo/data/flight-delays/flights.csv',
        help='Dataset file. Default: ./data/flight-delays/flights.csv')


    # a super parser for sanity checks
    super_parser = argparse.ArgumentParser(
        parents=[tree_inference_parser])

    # get arguments
    super_parser.parse_args()
    tree_inference_args, _ = tree_inference_parser.parse_known_args()

    return (tree_inference_args)
