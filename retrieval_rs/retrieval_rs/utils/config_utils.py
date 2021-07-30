'''
Collection of utils for configuring scripts

@author: shagup
'''

import argparse


def arg_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_params(args):
    params = []
    for arg in vars(args):
        params.append((arg, getattr(args, arg)))

    params.sort()
    print("SystemLog: ------------------ Parameters passed ------------------")
    for key, value in params:
        print("SystemLog:\t %s: %s" % (key, value))
    print("SystemLog:\n")
