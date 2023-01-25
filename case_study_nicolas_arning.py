"""
Script for the life science data scientist case study from capgemini engineering

Programm first does some data preprocessing and some data exploration and then
generates two models. One that predicts asthma from patient data and one that
predict the probability of an asthma attack within the next two days.

The script produces one pdf called case_study_plots.pdf that stores all figures
Also produces a pickle file for the asthma prediction called
case_study_model_finalised.p. The time series prediction gets added to the
original input table and put out as Predicted_data.csv

Written by Nicolas Arning

"""

import argparse
import umap
import time
import pickle
import matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn import metrics

from statsmodels.tsa.vector_ar.var_model import VAR
from tqdm import tqdm
from catboost import CatBoostClassifier

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredText


###############            DATA WRANGLING FUNCTIONS        ####################


def load_data(inf):
    """
    Function that loads in input table and transforms the categorical
    variables into numbers for easier averaging

    Arguments:
        inf - String containing file name of the table to be opened
    Returns:
        df - Dataframe containing the patient data per day
        df_per_patient - Dataframe containing the patient data median
            for each patient

    """
    df = pd.read_csv(inf)

    # Print some basic stats
    print("The Data has the following characteristics:")
    print(df.describe().round(3))

    # Replace stringts with unique numbers
    replace_dict = {
        "No": 0.0,
        "Yes": 1.0,
        "Female": 2.0,
        "Male": 3.0,
        }

    # Reverse the dictionary for back translation
    replace_dict_reverse = {entry: key for key, entry in replace_dict.items()}

    # Get the median of values for all patients
    df_per_patient = df.replace(replace_dict).groupby(["Patient_ID"]).median()

    # Translate categorical variables back
    df_per_patient.replace(replace_dict_reverse, inplace=True)
    return(df, df_per_patient)


###############               PLOTTING FUNCTIONS           ####################


def meta_plotter(df, pdf):
    """
    Meta plotting function that calls all other plotting functions and stores
    the plots in a pdf.

    Arguments:
        df - Dataframe containing the patient data averaged for each patient
        pdf - pdf object where all plots are stored

    Returns:
        pdf - pdf object where all plots are stored

    """
    # Pair plot (correlation between variables)
    pdf = plot_pair(df, pdf)

    # Violin plots
    df_violin, df_scaled = prep_violin(df)
    pdf = plot_violin(df_violin, pdf)

    # Umap plot
    pdf = plot_umap(df_scaled, pdf)

    return(pdf)


def individual_pair_plot(df):
    """
    Plots a seaborn pair plot that scatters every column against each other.
    coupled out of the other pairplot function so code doesn't have to be
    repeated for male and female.

    Arguments:
        df - Dataframe containing the patient data averaged for each patient

    Returns:
        fig - figure object containing the pairplot
    """

    sns.pairplot(
        df,
        hue='Asthma',
        palette=["lightcoral", "cornflowerblue"],
        diag_kws=dict(fill=False),
    )
    fig = plt.gcf()

    return(fig)


def plot_pair(df, pdf):
    """
    Plots seaborn pair plot of ever column against each other. Sits on top
    of individual_pair_plot function

    Arguments:
        df - Dataframe containing the patient data averaged for each patient
        pdf - pdf object where all plots are stored

    Returns:
        df - Dataframe containing the patient data averaged for each patient
    """

    # Drop columns that are not of interest for plotting
    df_pair = df.drop(["Day_ID", "Height_Cm", "AsthmaAttack"], axis=1)

    # Split the pairplots by gender
    fig = individual_pair_plot(df_pair[df_pair["Gender"] == "Male"])
    fig.suptitle("Pairplot Male")
    plt.tight_layout()
    pdf.savefig(fig)

    fig = individual_pair_plot(df_pair[df_pair["Gender"] == "Female"])
    fig.suptitle("Pairplot Female")
    plt.tight_layout()
    pdf.savefig(fig)

    return(pdf)


def prep_violin(df):
    """
    Prepare dataframe for seaborn violin plots as they need weird long form
    dataframes. Also scales dataframe for plotting

    Arguments:
        df - Dataframe containing the patient data averaged for each patient

    Returns:
        df_violin - Dataframe averaged over patient in long form
        df_scaled - Dataframe min max scaled 0 to 1

    """

    # We can ignore Day ID here
    df["Day_ID"] = df["Day_ID"].astype(object)

    # Get continual variables
    df_numerical = df.select_dtypes(include=[float])
    columns = df_numerical.columns

    # Scale per column so that max value is set to 1 and minimum to 0
    df_numerical = pd.DataFrame(
        MinMaxScaler().fit_transform(df_numerical),
        columns=columns)

    # Get categorical variables to be used later
    df_categorical = df.select_dtypes(include=["object"])
    replace_dict = {
        "Yes": 1.0,
        "No": 0.0,
        "Male": 1.0,
        "Female": 0.0,
        }
    df_categorical.replace(replace_dict, inplace=True)

    # Make the long for table necessary for seaborn violin plots
    df_violin = []
    for col in df_numerical.columns:
        df_violin += zip(df_numerical[col],
                         [col] * df_numerical.shape[0],
                         df["Asthma"],
                         df["Gender"])

    df_violin = pd.DataFrame(
        df_violin,
        columns=[
            "Value",
            "Measurement",
            "Asthma",
            "Gender"])

    # Get the scaled continual variables and the categorical variables back
    # together
    df_numerical.reset_index(drop=True, inplace=True)
    df_categorical.reset_index(drop=True, inplace=True)
    df_scaled = pd.concat([df_numerical, df_categorical], axis=1)

    return(df_violin, df_scaled)


def plot_violin(df, pdf):
    """
    Make seaborn violin plots.

    Arguments:
        df - Dataframe containing the patient data averaged for each patient
        pdf - pdf object where all plots are stored

    Returns:
        pdf - pdf object where all plots are stored
    """

    # Make two plots beneath each other
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    plt.title("Biomarkers split by having Asthma or not")

    # Make a male violin plot
    axes[0] = individual_violin_plots(df[df["Gender"] == "Male"], axes[0])
    axes[0].title.set_text('Male participants')

    # Make a female violin plot
    axes[1] = individual_violin_plots(df[df["Gender"] == "Female"], axes[1])
    axes[1].title.set_text('Female participants')

    plt.tight_layout()
    pdf.savefig(fig)

    return(pdf)


def individual_violin_plots(df_violin, ax):
    """
    Make individual violin plots. Seperate from plot_violin so code didn't have
    to be repeated.

    Arguments:
        df_violin - Dataframe averaged over patient in long form
        ax - matplotlib ax object

    Returns:
        ax - matplotlib ax object

    """

    sns.violinplot(
            data=df_violin,
            x="Measurement",
            y="Value",
            hue="Asthma",
            split=True,
            cut=0,
            bw=0.5,
            palette=["lightcoral", "cornflowerblue"],
            ax=ax
            )

    return(ax)


def plot_umap(df, pdf):
    """
    Plots a 2-D representation of the data using the Uniform Manifold
    Approximation and Projection (UMAP) algorithm.

    Arguments:
        df - Dataframe containing the patient data averaged for each patient
        pdf - pdf object where all plots are stored

    Returns:
        pdf - pdf object where all plots are stored

    """

    # Initialise a UMAP object
    reducer = umap.UMAP(
            n_neighbors=70,
            min_dist=0.5
            )

    # Fit to the data to create 2-D embedding
    embedding = reducer.fit_transform(df)

    # Feed the x and y coordinates of the embedding back into the dataframe
    df["Umap_x"] = embedding[:, 0]
    df["Umap_y"] = embedding[:, 1]

    # Make a scatterplot of the 2D-projection
    fig = plt.figure()
    sns.scatterplot(
        data=df,
        x="Umap_x",
        y="Umap_y",
        hue="Asthma",
        palette=["lightcoral", "cornflowerblue"],
        style="Gender",
        )

    plt.title("Two dimensional projection of Data split by Asthma and Gender")
    plt.tight_layout()
    pdf.savefig(fig)

    return(pdf)


###############         MACHINE LEARNING CLASSIFCATION     ####################


def meta_classification(df, pdf):
    """
    Function that calls all other functions necessary for doing the machine
    learning classification.

    Arguments:
        df - Dataframe containing the patient data averaged for each patient
        pdf - pdf object where all plots are stored

    Returns:
        pdf - pdf object where all plots are stored
    """

    # Recode the categorical cols again but this time not unique
    categorical_cols = df.select_dtypes(['object']).columns
    replace_dict = {
        "Yes": 1,
        "No": 0,
        "Male": 1,
        "Female": 0,
        }
    df.replace(replace_dict, inplace=True)
    df[categorical_cols] = df[categorical_cols].astype("category")

    # Train test splitting but patient aware
    df_train, df_test, label_train, label_test = patient_aware_data_splitting(
                                                                              df)

    # If a classifier is specified skip the hyper-parameter tuning
    if args.clf:
        model = pickle.load(open(args.clf, "rb"))
    else:
        # If not do hyper-parameter tuning
        model = hyperparameter_tuning(df_train, label_train, categorical_cols)
        pickle.dump(model, open("case_study_model_hyper_tuned.p", "wb"))

    feature_importance = zip(df_train.columns, model.get_feature_importance())
    feature_importance = sorted(
        feature_importance,
        key=lambda x: x[1],
        reverse=True)

    # Predict labels for scoring and plot the scoring
    label_predict = model.predict_proba(df_test)[:, 1]
    pdf = test_model(label_predict, label_test, feature_importance, pdf)

    # Use all the data (training and testing) to fit the model fully
    model = finalise_model(model, df)
    pickle.dump(model, open("case_study_model_finalised.p", "wb"))
    return(pdf)


def patient_aware_data_splitting(df):
    """
    Function that splits data into training and testing, but does so in a patient
    aware fashion, so data from the same patient doesn't land in both training
    and testing, which would lead to an overestimation of classification
    accuracy

    Arguments:
        df - Dataframe containing the patient data averaged for each patient

    Returns:
        df_train - Dataframe with training data
        df_test - Dataframe with testing data
        label_train - Training labels
        label_test - Testing labels
    """

    # Get all unique patient IDs
    all_patients = pd.Series(df["Patient_ID"].unique())
    test_set_size = int(len(all_patients)*0.25)
    # Sample 25% of the patients
    test_patients = all_patients.sample(test_set_size, random_state=1)

    # Split the dataframe in testing and training
    df_test = df[df["Patient_ID"].isin(test_patients)]
    df_train = df[~df["Patient_ID"].isin(test_patients)]

    # Split the labels in testing and training
    label_test = df_test["Asthma"]
    label_train = df_train["Asthma"]

    # Drop irrelevant columns
    cols2drop = ["Asthma", "Height_Cm", "Patient_ID", "Day_ID", "AsthmaAttack"]
    df_train.drop(columns=cols2drop, inplace=True)
    df_test.drop(columns=cols2drop, inplace=True)

    return(df_train, df_test, label_train, label_test)


def hyperparameter_tuning(df_train, label_train, categorical_cols):
    """
    Function that initialises and turns hyperparameters of catboost algorithm
    in a randomised gridsearch using a 5 fold stratified cross validation.

    Arguments:
        df_train - Dataframe with training data
        label_train - Training labels
        categorical_cols - Names of categorical columns

    Returns:
        model - catboostclassifier object
    """

    # Define Hyper-parameter space to be explored
    parameter_space = {
        "depth": np.linspace(4, 10, 6, dtype=int),
        "learning_rate": np.linspace(0.001, 0.5, 10, dtype=float),
        "iterations": np.linspace(10, 200, 10, dtype=int),
        "random_strength": np.linspace(1, 200, 10, dtype=int),
        "l2_leaf_reg": np.linspace(0, 0.5, 10, dtype=float),
    }

    # Use stratified 5 fold cross validation as the data-set is class
    # imbalanced
    split = StratifiedKFold(
        n_splits=5
        )

    # Get categorical columns as catboostclassifier needs this information
    categorical_cols = set(categorical_cols).intersection(
        set(df_train.columns))

    # Initialise the model
    model = CatBoostClassifier(
        objective="Logloss",
        thread_count=1,
        cat_features=list(categorical_cols),
        verbose=0
    )

    # Perform randomised gridsearch for 4000 iterations on 5 splits totalling
    # 2000 fits of the model
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameter_space,
        n_jobs=args.t,
        n_iter=2000,
        scoring="roc_auc",
        verbose=2,
        cv=split,
        random_state=0
    )
    search.fit(df_train, label_train)

    # Display the hyper-parameters
    print("Tuned hyper-parameters:")
    print(search.best_params_)

    model = search.best_estimator_.fit(df_train, label_train)

    return(model)


def test_model(label_predict, label_test, feature_importance, pdf):
    """
    Test the accuracy of the model and plot performance metrics and feature
    importance of model

    Arguments:
        label_predict - Predicted labels
        label_test - Testing labels
        feature_importance - List of tuples containing feature importance per
            column
        pdf - Pdf object where all plots are stored

    Returns:
        pdf - Pdf object where all plots are stored

    """
    # Round the predicted probabilities to full integers
    rounded_label_predict = np.rint(label_predict)

    # Get false positive, true positive rate for receiver-operator curve(ROC)
    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(
        label_test,  label_predict)

    # Get area under the ROC
    auc = metrics.roc_auc_score(label_test, label_predict)

    # Get accuracy
    accuracy = metrics.accuracy_score(label_test, rounded_label_predict)

    print("Here is a short accuracy report")
    print(metrics.classification_report(label_test, rounded_label_predict))

    # Plot ROC curve
    fig = plt.figure()
    sns.lineplot(
        x=false_positive_rate,
        y=true_positive_rate,
        )
    plt.plot([0, 1], [0, 1], ls="--", c="silver")
    label = "AUC = {:.2f}\n Accurary = {:.2f}".format(auc, accuracy)
    text_box = AnchoredText(label, frameon=True, loc=4, pad=0.5)
    plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    plt.gca().add_artist(text_box)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    pdf.savefig(fig)
    plt.close()

    # Plot feature importance as well
    columns, feature_importance = zip(*feature_importance)
    y_positions = list(range(len(columns)))
    fig = plt.figure()
    colors = [
        matplotlib.cm.get_cmap('Spectral')(x)
        for x in np.linspace(0, 1, len(feature_importance))]
    plt.barh(
        width=feature_importance,
        y=y_positions,
        alpha=0.5,
        color=colors
        )
    plt.yticks(ticks=y_positions, labels=columns)
    plt.gca().invert_yaxis()
    plt.xlabel("Split feature importance")
    plt.ylabel("Biomarkers")
    plt.tight_layout()
    pdf.savefig(fig)

    return(pdf)


def finalise_model(model, df):
    """
    Use the full data, not split into training and testing, to finalise the
    model and store it in a pickle object
    Arguments:
        model - Trained catboostclassifier model
        df - Dataframe containing the patient data averaged for each patient

    Returns:
        model - Finalised catboostclassifier model
    """

    labels = df["Asthma"]

    df.drop(columns=["Asthma",
                     "Patient_ID",
                     "Height_Cm",
                     "Day_ID",
                     "AsthmaAttack"],
            inplace=True
            )

    model.fit(df, labels)
    return(model)


###############            TIME SERIES PREDICTION          ####################


def time_series_prediction(df, pdf):
    """
    Function that trains a vector autregression (VAR) algorithm to predict
    asthma attack probability forward by two days for each patient

    Arguments:
        df - Dataframe containing the patient data per day
        pdf - pdf object where all plots are stored

    Returns:
        pdf - pdf object where all plots are stored

    """
    # Get only the patients with asthma as they are the only ones getting
    # asthma attacks
    df = df[df["Asthma"] == "Yes"]

    all_patients = list(df["Patient_ID"].unique())

    # Get the Day ID into pandas readable timedelta by adding to date of today
    df["Today"] = pd.Timestamp.today().date()
    df["Day_ID_new"] = df["Day_ID"].apply(lambda x: pd.Timedelta(x, unit='D'))
    df["Day_ID_new"] = df["Today"] + df["Day_ID_new"]

    # Now drop today again
    df = df.drop(columns=["Today"])

    # Initialise figure
    fig = plt.figure()
    x_true = range(11)
    x_pred = range(11, 13)
    ax = plt.gca()

    # Keep the columns interesting for predictions
    cols2keep4prediction = ["AsthmaAttack", "PEF", "FEV1", "FVC"]
    other_columns = list(set(df.columns) - set(cols2keep4prediction))
    new_df = []
    for patient in tqdm(all_patients):   # Loop through all patients

        # Get only biomarkers from patient
        df_patient = df[df["Patient_ID"] ==
                        patient].sort_values(by="Day_ID_new")

        # Set the Day ID as index
        df_patient.index = pd.DatetimeIndex(
            df_patient["Day_ID_new"], freq="infer")

        # Reduce dataframe
        df4prediction = df_patient[cols2keep4prediction]

        # Predict all relevant columns using VAR
        df_predict = make_time_series_prediction(df4prediction, 2)
        attack_pred = df_predict["AsthmaAttack"]

        # Plot of the predicted probability of an asthma attack over time
        sns.scatterplot(
            x=x_true,
            y=df_patient[-11:]["AsthmaAttack"],
            alpha=0.7,
            color="silver",
            marker="x",
            ax=ax
        )

        sns.scatterplot(
            x=x_pred,
            y=attack_pred,
            alpha=0.5,
            color="lightcoral",
            marker="x",
            ax=ax
        )

        # Now also get the other biomarkers
        df_predict[other_columns] = df_patient.loc[df_patient.index[0], other_columns]
        df_predict["Day_ID"] = [
            "predicted_{}".format(x) for x in range(
                df_predict.shape[0])]
        df_patient = pd.concat([df_patient, df_predict], axis=0)
        new_df.append(df_patient)

    # Plot a line where the prediction starts
    plt.axvline(10, alpha=0.7, ls="--", c="silver")
    pdf.savefig(fig)

    # Get all the old patient data and the predicted data together in a new
    # dataframe to write to file
    new_df = pd.concat(
        new_df,
        axis=0)
    new_df.drop(columns="Day_ID_new").round(3).to_csv(
        "Predicted_data.csv",
        index=False)

    return(pdf)


def make_time_series_prediction(df, steps):
    """
    Individual function for making time series prediction using VAR so the
    time_series_prediction function is cleaner

    Arguments:
        df - Dataframe containing the patient data per day
        steps - Integer of how many steps should be predicted into the future
    Returns:
        df_predict - Dataframe object where the prediction is added to the old
            values

    """

    # Initialise model
    model = VAR(endog=df)

    # Fit the model
    model_fit = model.fit()
    lag_order = model_fit.k_ar

    # Predict forward
    prediction = model_fit.forecast(df.values[-lag_order:], steps=steps)

    # Store prediction in dataframe
    df_predict = pd.DataFrame(prediction, columns=df.columns).abs()

    return(df_predict)


##############                MAIN FUNCTION                ####################

def main():
    """
    This is the main function
    """

    # Time the whole script
    start_time = time.time()

    # Read in command line arguments
    parser = argparse.ArgumentParser(
                    prog='Case study Nicolas Arning',
                    description=""" Builds two machine learning models for
                        analysing asthma from patient data """,
                    epilog='Please specify command line arguments')

    parser.add_argument('-i',
                        type=str,
                        help="Input table")
    parser.add_argument('-t',
                        type=int,
                        help="Number of threads")
    parser.add_argument('-clf',
                        default=False,
                        help="Instance of pre-trained-classifier")
    parser.add_argument('--skip_plotting',
                        action="store_true",
                        default=False,
                        help="Skip the plotting")
    parser.add_argument('--skip_classification',
                        action="store_true",
                        default=False,
                        help="Skip the Asthma classification")
    parser.add_argument('--skip_time_prediction',
                        action="store_true",
                        default=False,
                        help="Skip the Asthma attack prediction")

    global args
    args = parser.parse_args()

    # Read in table
    df, df_per_patient = load_data(args.i)

    # Initialise pdf document to store all plots in
    pdf = PdfPages('case_study_plots.pdf')

    # Check if plots for data explorations should be skipped
    if not args.skip_plotting:
        pdf = meta_plotter(df_per_patient, pdf)

    # Check if asthma classification should be skipped
    if not args.skip_classification:
        pdf = meta_classification(df.copy(), pdf)

    # Check if asthma attack prediction should be skipped
    if not args.skip_time_prediction:
        pdf = time_series_prediction(df, pdf)

    pdf.close()

    end_time = time.time()
    print("Elapsed Time: {:.2f}\nExiting.".format(end_time-start_time))


if __name__ == "__main__":
    main()
