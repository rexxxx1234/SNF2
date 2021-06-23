import pandas as pd


def filter_non_tumor(omics):
    omics = omics.filter(regex="01$", axis=0)
    return omics


def find_intersecting_sample(omics_list):
    original_order = [0] * (len(omics_list))
    subsample_list = []

    for i in range(0, len(omics_list)):
        original_order[i] = list(omics_list[i].index)

    common_list = original_order[0]
    for i in range(1, len(omics_list)):
        common_list = list(set(common_list).intersection(original_order[i]))
    print("number of intersecting samples: ", len(common_list))

    for i in range(0, len(omics_list)):
        subsample_list.append(omics_list[i].loc[common_list])

    return subsample_list
