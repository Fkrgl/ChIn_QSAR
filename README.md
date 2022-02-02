#---------------------------------------- Running istructions for QSAR model ------------------------------------------#

# The main function to make predictions with the model is in in the script qsar.py
# the script needs:
    -  pre processed training data in csv format to train the model (trainings_data.csv)
    -  test data in sdf format which should be predicted
    -  output path of the predictions of the test data

# program call:

    python src/qsar.py -in_train dat/trainings_data.csv -in_test PATH_TO_TEST_DATA -out_path PATH_TO_OUTPUT_FILE


# missing features:
if the test file contains compounds for which our pre-selected features result in NaN, then this compound is printed
to the console and excluded from the prediction
