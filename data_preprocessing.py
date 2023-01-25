"""
Script to preprocess data for doublethink
"""
import numpy as np
import pandas as pd
import argparse
import sys
import collections
import re
import sklearn

from multiprocessing import Pool

import sklearn.neighbors._base
# Missforest needs this
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest


def load_data(inf):
    """
    Function that reads in data from multiple different formats
    """
    # Determine what type the file is and read in accordingly
    file_ending = inf.split(".")[-1]
    read_in_dict = {
        "csv": pd.read_csv,
        "tsv": pd.read_table,
        "h5": pd.read_hdf,
        "hdf": pd.read_hdf,
        "xls": pd.read_excel,
        "xlsx": pd.read_excel,
        "json": pd.read_json,
        "p": pd.read_pickle,
        "pickle": pd.read_pickle,
    }
    # See if it works
    try:
        df = read_in_dict[file_ending](inf)
    except KeyError:  # if not throw error
        print("""ERROR.
              Could not read file extension, or file format not known.
              These the file extensions I can read: .{}.
              EXITING""".format(", .".join(read_in_dict.keys)))
        sys.exit()
    # If row names are given throw it out
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    return(df)


def encode_df(df, column_encoding):
    """
    Function that encodes the data
    """
    # Read data encoding table
    ce_df = pd.read_table(column_encoding, sep=",")
    # Turn into dict
    column_encoding_dict = dict(zip(ce_df.iloc[:, 0], ce_df.iloc[:, 1]))
    for col in df.columns:
        if col not in column_encoding:
            if col == args.id:  # if its the id column skip
                continue
            print(
                """WARNING. The column {} was not found in your column encoding
                  file. The column can therefore not be properly encoded
                  """.format(col))
            continue
        # Encode data
        encoding = column_encoding_dict[col]
        if encoding == "factor":
            df[col] = df[col].astype("category")
        elif encoding == "integer":
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype(encoding)
    # Return the dataframe and the encoding. We need the encoding later for
    # averaging
    return(df, column_encoding_dict)


def guess_encoding_df(df):
    """
    Function that guesses columns encoding if none was specified
    """
    encoding_dict = {}
    for col in df:
        tmp_col = df[col].fillna(0).astype(float)
        # If they are all integers
        if all(x.is_integer() for x in tmp_col):
            # Find out if less than 15 different numbers or number in name
            if tmp_col.nunique() > 15 or "number" in col.lower():
                encoding_dict[col] = "integer"
                df[col] = df[col].astype(float)
            else:  # if not probably a factor?
                encoding_dict[col] = "factor"
                df[col] = df[col].astype("category")
        else:  # if it s not an integer encode as float
            encoding_dict[col] = "float"
            df[col] = df[col].astype(float)

    # Write this out to file so user can go through and change where necessary
    with open("guess_column_encoding.csv", "w+") as outf:
        for col, encoding in encoding_dict.items():
            outf.write("{},{}\n".format(col, encoding))

    return(df, encoding_dict)


def average_function(input_tuple):
    """
    Function to average slices of the dataframe with
    """
    base_name, df_slice, col_type = input_tuple
    # Do different averages depending on type of column
    if col_type == "float":
        averaged_col = df_slice.mean(axis=1, skipna=True)
    elif col_type == "integer":
        averaged_col = df_slice.median(axis=1, skipna=True).round(decimals=0)
    else:
        averaged_col = df_slice.mode(axis=1, dropna=True)
        # Mode can be two different values if the same two avalues are used
        # the same amount of times. If so just choose the first.
        if len(averaged_col.shape) > 1:
            averaged_col = averaged_col.iloc[:, 0]
    # return the name and the column
    return(base_name, averaged_col)


def average_cols(df, encoding_dict):
    """
    Average biomarkers that have multiple measurements
    """
    # Multiple measurements are seperated by dot at the end of the name
    base_name_list = [col.split(".")[0] for col in df.columns]

    # Base names being double indicates multiple measurements
    base_name_double = [item for item, count in collections.Counter(
                            base_name_list).items() if count > 1]
    base_name_list = np.array(base_name_list)
    col_array = np.array(df.columns)

    input_list = []  # this is the list that goes into the averaging function
    remove_list = []  # those that have been averaged can be removed
    if len(base_name_double) == 0:  # If there's nothing to average just exit
        return(df)

    for base_name in base_name_double:
        if base_name.startswith(
                ("CHARLSON", "ICD")):  # disease codes have been pre-averaged
            continue

        # Get all columns that are double
        columns = list(col_array[base_name_list == base_name])

        # Collect all ingredients for the avaeraging functions
        input_list.append((base_name, df[columns], encoding_dict[columns[0]]))
        remove_list += columns

    # Open pool of threads
    pool = Pool(args.t)

    # Average in parallel
    average_list = pool.map(average_function, input_list)

    # Parse output
    headers, averaged_columns = zip(*average_list)

    # Make averaged columns into one dataframe
    new_cols = pd.concat(averaged_columns, axis=1, ignore_index=True)
    new_cols.columns = headers

    # Drop unaveraged columns from old dataframe
    df.drop(columns=remove_list, inplace=True)

    # Merge old dataframe and new averaged columns
    df = pd.concat([df, new_cols], axis=1)
    df.columns = [x.split(".")[0] for x in df.columns]

    # Now drop NAs
    thresh = df.shape[0] * (1-args.na)
    df.dropna(thresh=thresh, axis=1, inplace=True)

    return(df)


def impute_df(df):
    """
    Function that imputes data. This takes a looooong time
    """

    imputer = MissForest(n_jobs=args.t,
                         max_iter=5,
                         verbose=1,
                         n_estimators=50
                         )
    df_imputed = imputer.fit_transform(df)
    return(df)


def encode_dummies(df):
    """
    Encode factors into dummy variables
    """
    cols_to_dummify = []
    # get all categorical columns
    categorical_cols = list(df.select_dtypes(include=['category']).columns)
    for col in categorical_cols:
        if df[col].nunique() > 2:  # only necessary to dummify if more than 2 levels
            print("Dummifying factor: {} into {} dummy columns".format(
                col, df[col].nunique()-1))
            cols_to_dummify.append(col)

    # Get dummy variables
    df = pd.get_dummies(df,
                        columns=cols_to_dummify,
                        prefix="dummy",
                        drop_first=True,
                        dtype=int)

    # Filter low number levels
    print("Filtering dummy columns with less than {} prevalence".format(args.dt))
    threshold_value = int(args.dt*df.shape[0])
<<<<<<< HEAD
    dummy = [i for i in df.columns if i.startswith('dummy')]
    cols2drop = []
    for col in dummy:
=======
    dummy_cols = []
    dummy = [i for i in df.columns if i.startswith('dummy')]
    cols2drop = []
    for col in dummy_cols:
>>>>>>> main
        if df[col].sum() < threshold_value:
            cols2drop.append(col)
    print("The following columns will be dropped: {}".format(",".join(cols2drop)))
    df.drop(columns=cols2drop, inplace=True)

    return(df)


if __name__ == "__main__":
    """
    main function
    """

    # Read in command line arguments
    parser = argparse.ArgumentParser(
        description="Do pre-processing for doublethink")
    parser.add_argument(
        "-i",
        type=str,
        help="Input file")
    parser.add_argument(
        "-ce",
        type=str, default=False,
        help="""Csv that is composed of the column and its type
        (float, integer, factor) seperated by a comma, if you do not specify
        a file here the script will try to guess the column type""")
    parser.add_argument(
        "-id",
        type=str, default="eid",
        help="Identifier column of your file")
    parser.add_argument(
        "-t",
        type=int, default=1,
        help="How many threads to use")
    parser.add_argument(
        "-na",
        type=float, default=0.1,
        help="Threshold for cutting NAs")
    parser.add_argument(
        "-dt",
<<<<<<< HEAD
        type=float, default=0.001,
=======
        type=float, default=0.01,
>>>>>>> main
        help="Prevalence under which dummies should be cut")
    parser.add_argument(
        "--impute", action='store_true',
        help="""Use machine learning imputation of missing values with MissForest.
        WARNING this will take a LONG time.
        Imputing the whole UK Biobank took 3 weeks on 20 threads""")
    global args
    args = parser.parse_args()

    # Give the whole shabang a name
    output_name = args.i.split(".")[0]

    print("Loading data")
    df = load_data(args.i)  # load input table

    if args.id != "eid":
        df.rename(columns={args.id: "eid"}, inplace=True)  # rename id column

    print("Encoding data")
    if args.ce:  # See if encoding table was given
        df, encoding_dict = encode_df(df, args.ce)
    else:  # if not guess the encoding
        print(
            """WARNING, No column encoding specified, this script will try to
            guess the encoding please check the resulting file and make
            amendments wherever necessary, then feed the resulting file
            back to the script with the -ce argument\n
            This should also be included in the config file under
            columns_filename""")
        df, encoding_dict = guess_encoding_df(df)

    # Average over multiple measurements
    print("Averaging over multiple measurements")
    df = average_cols(df, encoding_dict)

    # Encode dummy variables
    print("Encoding dummies")
    df = encode_dummies(df)

    # If impute flag was given, impute missing values. This takes forever
    if args.impute:
        df = impute_df(df)

    # Take all weird characters out of column headers
    rename_dict = {}
    for col in df.columns:
        col_name = re.sub(
            r'([^\s\w])+',
            '',
            col.split(".")[0].replace(
                " ",
                "_").replace(
                "-",
                "_").replace(
                "___",
                "_").replace(
                "_/_",
                "_").replace(
                "/",
                "_"))
        rename_dict[col] = col_name
    df.rename(columns=rename_dict, inplace=True)

    # Write to file
    print(
        "Dropping preprocess data in {}.processed4doublethink.csv".format(
            output_name))
    df.to_csv("{}.processed.csv".format(output_name))

    # Print out some baseline statistics
    print(
        "Dropping base statistics of input data in {}_description.tsv".format(
            output_name))
<<<<<<< HEAD
    df.describe().round(4).to_csv("{}_description.tsv".format(output_name), sep="\t")
=======
    df.describe().round(2).to_csv("{}_description.tsv".format(output_name), sep="\t")
>>>>>>> main
    print("Exiting")
