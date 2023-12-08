import pandas as pd
pd.set_option('mode.chained_assignment', None) #Suppresses SettingWithCopyWarning for now
import numpy as np
import os
import time
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from xgboost import XGBRegressor, XGBClassifier
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Assume that main.py is in same location as the Test and Train .csv files
if __name__ == '__main__':

    # Function to clean the input dataset.
    def cleanData(data, type, target) :
        # data = dataframe containing Training Data or Test Data
        # type = "Training" or "Test"
        # target = "Trim" or "Price"

        # First thing to do is look at our data for simple cleaning.
        # As a data scientist, need to find a balance of which variables to keep.
        # Many categorical variables, need to deal with them since some have dozens or more unique values.
        # In the interest of this exercise, an overview of my cleaning method is as follows:
            # Drop ListingID (just an id field, not be useful in prediction, rely on native dataframe index if needed)
            # Drop SellerListSrc (name of seller source likely doesn't matter)
            # Drop SellerCity (too granular; use SellerState instead for geographical variation)
            # Drop SellerIsPriv (only about 0.2% of our data has this value = True: trainData['SellerIsPriv'].sum()/len(trainData) = 0.0022)
            # Drop SellerName (I did a quick check that each SellerName should have the same SellerRating, so these are basically redundant variables--plus I assume through domain assumption that a SellerRating somewhat accurately captures the value of the dealership's name brand)
            # Drop SellerZip (too granular; SellerState instead for geographical variation)
            # Drop VehBodystyle (quick check shows that all are SUV, so this isn't needed)
            # Replace VehColorExt with Primary Colors only
            # Replace VehColorInt with Primary Colors only
            # Replace VehEngine with 3.6, 5.7, 6.4, or "Other" (most common engine sizes)
            # Convert VehFeats into a Count of Features (proxy; otherwise too much text to parse for scope of this exercise--would do text analysis in actual business setting)
            # Scrape VehHistory to get # of Owners (Integer), Buyback Protection Eligible (Boolean), Accidents Reported (Boolean)
            # Drop VehModel (quick check shows that all Cadillac are XT5, and all Jeep are Grand Cherokee, so this is redundant)
            # Drop VehSellerNotes (not parsing text for this exercise due to time/scope)
            # Drop VehType (all cars are Used, no variation here)
            # Drop VehTransmission (all cars seem to be 8-Speed or provide minimal information, domain knowledge says this probably not useful)
        # For this exercise, I went through each variable and used, e.g., trainData['VARNAME'].unique() or .nunique() to see how many unique values were in VARNAME to make the above decisions.
        # However, there are Python packages which get summary statistics of each column that are nice to use as well!
        # Nota Bene: I don't include asserts for defensive programming due to time/scope/readability, but I typically include some basic asserts in my code for good practices.

        # Training Data has a few cars without SellerListSrc. Turns out that a lot of information is missing in the other columns too (i.e., Engine, Fuel, etc.)
        # So, remove those rows without SellerListSrc since they are missing too much data across the row to impute.
        # I remove SellerListSrc regardless, since I don't think it will be useful (domain knowledge) in model prediction.
        data = data[data["SellerListSrc"].notna()]
        data.drop(columns = ["SellerListSrc", "SellerCity", "SellerIsPriv", "SellerName", "SellerZip",
                             "VehBodystyle", "VehModel", "VehSellerNotes", "VehType", "VehTransmission"], inplace = True)

        # Keep the ListingID if Test Data (need to use it to merge back output from applying model)
        if type == "Training" :
            data.drop(columns = ["ListingID"], inplace = True)

        # Some rows don't have Mileage info. Replace with median for simple imputation for this exercise's scope.
        # I could also consider alternative imputation (e.g., MICE, Regression, or KNN).
        data.fillna(value={"VehMileage": np.floor(data.VehMileage.astype(float).mean())}, inplace=True)

        ######################### PARSE VEHICLE CHARACTERISTICS #########################

        # Colors: White, Black, Silver, Blue, Red, Gray, Green, Other; for both Ext and Int colors.
        # If a vehicle has multiple colors listed, ties are broken in color listing order (simple rule for scope).
        def parseColors(df) :
            colors = ["WHITE", "BLACK", "SILVER", "BLUE", "RED", "GRAY", "GREEN"]
            df["VehColorExt"] = df["VehColorExt"].str.upper()
            df["VehColorInt"] = df["VehColorInt"].str.upper()
            for c in colors :
                df.loc[df["VehColorExt"].str.contains(c, na = False), "VehColorExt"] = c
                df.loc[df["VehColorInt"].str.contains(c, na = False), "VehColorInt"] = c
            df.loc[~df["VehColorExt"].isin(colors), "VehColorExt"] = "OTHER"
            df.loc[~df["VehColorInt"].isin(colors), "VehColorInt"] = "OTHER"
            return df
        data = parseColors(data)

        # Vehicle Drive
        # There are a few that list AWD or 4WD. Default to AWD.
        def parseDrive(df) :
            df["VehDriveTrain"] = df["VehDriveTrain"].str.upper()
            driveDict = {"4X4":"4WD", "4X4/4WD":"4WD", "4x4":"4WD", "FOUR WHEEL DRIVE":"4WD", "FRONT-WHEEL DRIVE":"FWD",
                         "ALL WHEEL DRIVE":"AWD", "ALL-WHEEL DRIVE WITH LOCKING AND LIMITED-SLIP DIFFERENTIAL":"AWD",
                         "AWD OR 4X4":"AWD", "ALL-WHEEL DRIVE":"AWD", "FRONT WHEEL DRIVE":"FWD", "4X4/4-WHEEL DRIVE":"4WD",
                         "ALL WHEEL" : "AWD", "ALLWHEELDRIVE" : "AWD", "4WD/AWD" : "AWD", "2WD" : "FWD"}
            df.replace({"VehDriveTrain" : driveDict}, inplace = True)
            df.fillna(value = {"VehDriveTrain" : "Missing"}, inplace=True) #Fill NaN with "Missing" string
            return df
        data = parseDrive(data)

        # Vehicle Engine
        def parseEngine(df) :
            engine = ["3.6", "5.7", "6.4"]
            for e in engine:
                df.loc[df["VehEngine"].str.contains(e, na=False), "VehEngine"] = e
            df.loc[~df["VehEngine"].isin(engine), "VehEngine"] = "Other"
            df.fillna(value = {"VehEngine" : "Missing"}, inplace = True) #Fill NaN with "Missing" string
            return df
        data = parseEngine(data)

        # Vehicle Features:
        # Could do some text analysis but that's a little beyond the scope of this exercise.
        # Instead, will replace VehFeats with a count of the number of features (count features as separated by comma) in each row as a proxy.
        # I also see that most of the values in this column are 8, so fill in NaN with 8 (median/mode value).
        def parseFeatures(df) :
            df["VehFeats"] = df["VehFeats"].str.count(',') + 1 #1 Comma = 2 Items, 2 Commas = 3 Items; default 0 commas = 1 item.
            df.fillna(value = {"VehFeats" : 8}, inplace = True)
            return df
        data = parseFeatures(data)

        # Vehicle Fuel:
        # Gasoline, E84 Flex Fuel, Diesel, and Unknown. Overwhelmingly Gasoline.
        # Change to boolean: isGasoline
        def parseFuel(df) :
            df["VehFuel"] = df["VehFuel"].str.upper()
            df["isGasoline"] = df["VehFuel"].str.contains("GASOLINE")
            df.insert(df.columns.get_loc("VehFuel") + 1, "isGasoline", df.pop("isGasoline"))
            df.drop(columns = "VehFuel", inplace = True)
            return df
        data = parseFuel(data)

        # Vehicle History:
        # Extract (use Regex) into new Variables: # of Owners, whether Buyback Protection Eligible (Boolean), and Accident Reported (Boolean).
        def parseVehHistory(df) :
            historyList = ["Owners", "Buyback", "Accident"]
            df["VehHistory"] = df["VehHistory"].str.upper()
            for h in historyList:
                if h == "Owners" :
                    df["VehOwners"] = df["VehHistory"].str.extract(r"(\d+) OWNER[S]?")
                    df.insert(df.columns.get_loc("VehHistory") + 1, "VehOwners", df.pop("VehOwners"))
                else :
                    colName = "Veh" + h
                    df[colName] = df["VehHistory"].str.contains(h.upper())
                    df.insert(df.columns.get_loc("VehHistory") + 1, colName, df.pop(colName))
            df.drop(columns = "VehHistory", inplace = True)
            # Handle Missing Values:
            #   Accident (Boolean): Majority class is False
            #   Buyback (Boolean): Majority class is True
            #   Owners (Integer): Majority "class" is 1 Owner; Mean is 1.02 Owners --> Replace with 1 Owner.
            df.fillna(value={"VehAccident" : False, "VehBuyback" : True, "VehOwners" : 1}, inplace=True)
            df["VehOwners"] = pd.to_numeric(df["VehOwners"])
            return df
        data = parseVehHistory(data)

        # VehiclePriceLabel:
        # Good, Great, Fair, and Unknown categories
        def parsePriceLabel(df) :
            labelList = ["Good", "Great", "Fair"]
            df["VehPriceLabel"] = df["VehPriceLabel"].str.upper()
            for l in labelList :
                df.loc[df["VehPriceLabel"].str.contains(l, na=False), "VehPriceLabel"] = l
            df.fillna(value={"VehPriceLabel" : "UNKNOWN"}, inplace = True)
            return df
        data = parsePriceLabel(data)

        ######################### PARSE VEHICLE TRIM #########################

        # Since Vehicle_Trim is the dependent variable, if missing then drop.
        # We also won't use the Dealer_Listing_Price since it's implied that we can only use the other 26 variables.
        # Parsing Trim and Price is for Training Data only (since they're our target variables and not provided in Test Data!)
        if type == "Training" : # If cleaning Training Data
            if target == "Trim" : # If building Trim Model, drop the Price column
                data_Trim = data.dropna(subset = ["Vehicle_Trim"]).copy()
                data_Trim.drop(columns="Dealer_Listing_Price", inplace=True)

                # I notice that Trims are different by Vehicle Make: Cadillacs have specific Trims, and Jeeps have others.
                # So, build two models: one for Jeeps, one for Cadillacs.
                # For Jeeps, the Trims are: Laredo, Altitude, Limited, Overland, SRT, Summit, Trailhawk, Trackhawk, Upland, Sterling.
                #   Google tells me that the 75th Anniversary is a Laredo model: https://www.thebestchrysler.com/grand-cherokee-trim-levels-explained/
                #   Wikipedia (https://en.wikipedia.org/wiki/Jeep_Grand_Cherokee_%28WK2%29) says that:
                #       Upland -> Laredo
                #       Sterling -> Limited
                #       Trackhawk -> SRT
                # For Cadillacs, the Trims can be grouped into: Base, Luxury, Platinum, Premium.
                def parseTrim(df) :
                    df["Vehicle_Trim"] = df["Vehicle_Trim"].str.upper()
                    cadillacTrims = ["PREMIUM", "PLATINUM", "LUXURY"] #"Premium Luxury" Trim exists, so we need to replace "Premium" first!
                    for c in cadillacTrims :
                        df.loc[df["Vehicle_Trim"].str.contains(c, na=False), "Vehicle_Trim"] = c
                    df.loc[df["Vehicle_Trim"].str.contains("FWD", na=False), "Vehicle_Trim"] = "BASE"
                    jeepTrims = ["LAREDO", "ALTITUDE", "LIMITED", "OVERLAND", "SRT", "SUMMIT", "TRAILHAWK"]
                    for j in jeepTrims :
                        df.loc[df["Vehicle_Trim"].str.contains(j, na=False), "Vehicle_Trim"] = j
                    df.loc[df["Vehicle_Trim"].str.contains("ANNIVERSARY", na=False), "Vehicle_Trim"] = "LAREDO"
                    df.loc[df["Vehicle_Trim"].str.contains("UPLAND", na=False), "Vehicle_Trim"] = "LAREDO"
                    df.loc[df["Vehicle_Trim"].str.contains("STERLING", na=False), "Vehicle_Trim"] = "LIMITED"
                    df.loc[df["Vehicle_Trim"].str.contains("TRACKHAWK", na=False), "Vehicle_Trim"] = "SRT"
                    return df
                data = parseTrim(data_Trim)

                # To run our models, we need to get Dummy Variables of the Categorical Variables:
                #   SellerState, VehColorExt, VehColorInt, VehDriveTrain, VehEngine, VehPriceLabel
                # drop_first = True to avoid multicollinearity.
                data_Trim = pd.get_dummies(data_Trim, columns = ["SellerState", "VehColorExt", "VehColorInt",
                                                                 "VehDriveTrain", "VehEngine",
                                                                 "VehPriceLabel"], drop_first = True)

                # Now, let's split the data into Jeeps and Cadillacs for two separate models.
                jeepTrim = data_Trim[data_Trim["VehMake"] == "Jeep"].copy()
                jeepTrim.drop(columns = "VehMake", inplace = True)
                cadillacTrim = data_Trim[data_Trim["VehMake"] == "Cadillac"].copy()
                cadillacTrim.drop(columns = "VehMake", inplace = True)
                return jeepTrim, cadillacTrim
            else : # else, building the Price Model--less additional cleaning needed!
                data_Price = data.dropna(subset = ["Dealer_Listing_Price"]).copy() #Drop NA's in Price
                data_Price.drop(columns = "Vehicle_Trim", inplace = True) #Don't use Trim
                data_Price = pd.get_dummies(data_Price, columns = ["SellerState", "VehColorExt",
                                                                   "VehColorInt", "VehDriveTrain",
                                                                   "VehEngine", "VehPriceLabel"], drop_first = True)
                # Since I build one model for both Jeep and Cadillacs for Price, can use "VehMake" as a variable in model.
                data_Price["isJeep"] = data_Price["VehMake"] == "Jeep" #Convert VehMake into Boolean isJeep
                data_Price.insert(data_Price.columns.get_loc("VehMake") + 1, "isJeep", data_Price.pop("isJeep"))
                data_Price.drop(columns=["VehMake"], inplace = True)
                return data_Price
        else : # else, clean the Test Data.
            if target == "Trim" :
                data = pd.get_dummies(data, columns = ["SellerState", "VehColorExt", "VehColorInt", "VehDriveTrain",
                                                       "VehEngine", "VehPriceLabel"], drop_first = True)
                jeepTrim = data[data["VehMake"] == "Jeep"].copy()
                jeepTrim.drop(columns = "VehMake", inplace = True)
                cadillacTrim = data[data["VehMake"] == "Cadillac"].copy()
                cadillacTrim.drop(columns = "VehMake", inplace = True)
                return jeepTrim, cadillacTrim
            else :
                data_Price = pd.get_dummies(data, columns=["SellerState", "VehColorExt", "VehColorInt", "VehDriveTrain",
                                                           "VehEngine", "VehPriceLabel"], drop_first=True)
                data_Price["isJeep"] = data_Price["VehMake"] == "Jeep"  # Convert VehMake into Boolean isJeep
                data_Price.insert(data_Price.columns.get_loc("VehMake") + 1, "isJeep", data_Price.pop("isJeep"))
                data_Price.drop(columns = ["VehMake"], inplace = True)
                return data_Price

    ################## TRAIN VEHICLE_TRIM MODELS (CLASSIFICATION) #####################

    trainData = pd.read_csv(os.path.join(os.getcwd(), "Training_Dataset.csv"))  # Load in Training Dataset
    jeepTrain, cadillacTrain = cleanData(trainData, "Training", "Trim")

    # Run Random Forest, XGBoost, and (Multinomial) Logistic Regression for Classification.
    # Use ROC AUC as scoring criterion.
    # Create a Train-Test Split (with a random seed for replication; could think about stratifying--not used here due to time constraints).
    # Employ Cross-Validation and basic Grid Search for training and hyperparameter tuning.

    # Random Forest Classifier
    def rfModel(df, type) :
        # df = jeepTrain or cadillacTrain
        # type = "Jeep" or "Cadillac"
        df = df.copy() #avoids changing the actual underlying dataset
        vehicle_y = df.pop("Vehicle_Trim")
        start_rf = time.time()
        print(type)
        print("Starting Random Forest Grid Search")
        train_x, test_x, train_y, test_y = train_test_split(df, vehicle_y, test_size = 0.2, random_state = 322)
        param_grid = {'n_estimators': [10, 25, 50, 100], 'max_depth' : [5, 10, 15, 20]}
        rf_model = RandomForestClassifier(random_state = 322)
        grid_rf_model = GridSearchCV(rf_model, param_grid, cv = 10, n_jobs = 4, scoring = "accuracy")
        grid_rf_model.fit(train_x, train_y)
        print("Done with Random Forest Grid Search")
        end_rf = time.time()
        print(time.strftime("%Hh%Mm%Ss", time.gmtime(end_rf - start_rf)))
        rf_predict = grid_rf_model.predict(test_x)
        accuracy = (rf_predict == test_y).sum() / len(test_y)
        print("Accuracy: " + str(accuracy))
        rf_predict_prob = grid_rf_model.predict_proba(test_x)
        print("ROC AUC: " + str(metrics.roc_auc_score(test_y, rf_predict_prob, multi_class='ovr', average='weighted')))
        print("Best Parameters: ", grid_rf_model.best_params_)
        return grid_rf_model

    # XGBoost Classifier
    def xgbModel(df, type) :
        # df = jeepTrain or cadillacTrain
        # type = "Jeep" or "Cadillac"
        df = df.copy() #avoids changing the actual underlying dataset
        vehicle_y = df.pop("Vehicle_Trim")
        start_rf = time.time()
        print(type)
        print("Starting XGBoost Grid Search")
        le = LabelEncoder()
        vehicle_y_transform = le.fit_transform(vehicle_y)
        train_x, test_x, train_y, test_y = train_test_split(df, vehicle_y_transform, test_size = 0.2, random_state = 322)
        param_grid = {'n_estimators': [10, 25, 50, 100], 'max_depth' : [3, 5, 7, 9]}
        xgb_model = XGBClassifier(random_state = 322)
        grid_xgb_model = GridSearchCV(xgb_model, param_grid, cv = 10, n_jobs = 4, scoring = 'roc_auc_ovr') #multiclass one-versus-rest ROC AUC, inspired by instructions
        grid_xgb_model.fit(train_x, train_y)
        print("Done with XGBoost Grid Search")
        end_rf = time.time()
        print(time.strftime("%Hh%Mm%Ss", time.gmtime(end_rf - start_rf)))
        xgb_predict = grid_xgb_model.predict(test_x)
        accuracy = (xgb_predict == test_y).sum() / len(test_y)
        print("Accuracy: " + str(accuracy))
        xgb_predict_prob = grid_xgb_model.predict_proba(test_x)
        print("ROC AUC: " + str(metrics.roc_auc_score(test_y, xgb_predict_prob, multi_class = 'ovr', average = 'weighted')))
        print("Best Parameters: ", grid_xgb_model.best_params_)

        # Some legacy ConfusionMatrix Code could use in the future
        labels = le.classes_
        # cm = confusion_matrix(test_y, prediction)
        # ConfusionMatrixDisplay.from_predictions(test_y, prediction, display_labels=labels, xticks_rotation = "vertical")
        return grid_xgb_model, labels

    # Multinomial Logistic Regression
    def lrModel(df, type) :
        # df = jeepTrain or cadillacTrain
        # type = "Jeep" or "Cadillac"
        df = df.copy()  # avoids changing the actual underlying dataset
        vehicle_y = df.pop("Vehicle_Trim")
        start_rf = time.time()
        print(type)
        print("Starting Multinomial Logistic Regression")
        train_x, test_x, train_y, test_y = train_test_split(df, vehicle_y, test_size = 0.2, random_state = 322)
        lr = LogisticRegression(multi_class = "multinomial", solver = "newton-cg", max_iter = 200)
        lr.fit(train_x, train_y)
        print("Done with Multinomial Logistic Regression")
        end_rf = time.time()
        print(time.strftime("%Hh%Mm%Ss", time.gmtime(end_rf - start_rf)))
        prediction = lr.predict(test_x)
        accuracy = (prediction == test_y).sum() / len(test_y)
        lr_predict_prob = lr.predict_proba(test_x)
        print("Multinomial Logistic Regression Accuracy: " + str(accuracy))
        print("ROC AUC: " + str(metrics.roc_auc_score(test_y, lr_predict_prob, multi_class='ovr', average='weighted')))
        return lr

    ### Run the Models! ###
    # jeepTrain.Vehicle_Trim.value_counts(normalize = True) # Baseline is to predict Majority Class ("LIMITED"), would give 48.26%.
    jeepRF = rfModel(jeepTrain, "Jeep") #~10 seconds, 0.6514 Accuracy, 0.8463 ROC AUC
    jeepXGB, jeepLabels = xgbModel(jeepTrain, "Jeep") #~17 seconds, 0.6663 Accuracy, 0.8469 ROC AUC
    jeepLR = lrModel(jeepTrain, "Jeep") #~8 seconds, 0.5881 Accuracy, 0.7640 ROC AUC

    # cadillacTrain.Vehicle_Trim.value_counts(normalize = True) # Baseline is to predict Majority Class ("PREMIUM"), would give 42.46%.
    cadillacRF = rfModel(cadillacTrain, "Cadillac") #~8 seconds, 0.7105 Accuracy, 0.8841 ROC AUC
    cadillacXGB, cadillacLabels = xgbModel(cadillacTrain, "Cadillac") #~9 seconds, 0.7239 Accuracy, 0.8870 ROC AUC
    cadillacLR = lrModel(cadillacTrain, "Cadillac") #~2 seconds, 0.5925 Accuracy, 0.7602 ROC AUC

    # I see that XGBoost provides the best performance both in Accuracy and AUC_ROC, so we use it on the Test Data.
    # Need to make a Test Data pipeline since need to use ListingID to match Test Data input to Prediction output!
    testData = pd.read_csv(os.path.join(os.getcwd(), "Test_Dataset.csv"))
    jeepTest, cadillacTest = cleanData(testData, "Test", "Trim")

    jeepID = jeepTest.pop("ListingID")
    cadillacID = cadillacTest.pop("ListingID")

    # jeepTest might not have all the Dummy Variables if some categories were missing!
    test_cols = jeepTest.columns
    train_cols = jeepTrain.columns #Jeep and Cadillac datasets have same columns, so using one of the two is fine.
    set(test_cols) - set(train_cols) #empty; all Train columns are in Test
    missing = list(set(train_cols) - set(test_cols)) #there's some train columns not in test!
    missing.remove("Vehicle_Trim") #Don't need Vehicle_Trim (it messes up applying the model since it doesn't see Vehicle_Trim in the train data's X matrix)
    for m in missing :
        jeepTest[m] = False
        cadillacTest[m] = False
    colOrder = train_cols.to_list()
    colOrder.remove("Vehicle_Trim")

    # Re-order columns to match model fit order, so model fit can run properly.
    jeepTest = jeepTest[colOrder]
    cadillacTest = cadillacTest[colOrder]
    jeepOut = pd.Series(jeepXGB.predict(jeepTest), name = "Vehicle_Trim")
    cadillacOut = pd.Series(cadillacXGB.predict(cadillacTest), name = "Vehicle_Trim")

    # Dataframe Manipulation to obtain output in the desired format.
    jeepID = jeepID.reset_index(drop = True)
    jeepOut = jeepOut.reset_index(drop = True)
    jeepConcat = pd.concat([jeepID, jeepOut], axis=1)
    jeep_dict = dict(zip(range(len(jeepLabels)), jeepLabels))
    jeepConcat.replace({"Vehicle_Trim": jeep_dict}, inplace=True)

    cadillacID = cadillacID.reset_index(drop = True)
    cadillacOut = cadillacOut.reset_index(drop = True)
    cadillacConcat = pd.concat([cadillacID, cadillacOut], axis = 1)
    cadillac_dict = dict(zip(range(len(cadillacLabels)), cadillacLabels))
    cadillacConcat.replace({"Vehicle_Trim" : cadillac_dict}, inplace = True)

    trimOut = pd.concat([jeepConcat, cadillacConcat], ignore_index = True)
    trimOut = trimOut.sort_values(["ListingID"], ascending = True).reset_index(drop = True) #Final Classification (Vehicle_Trim) Test Output

    ################## TRAIN DEALER_LISTING_PRICE MODELS (REGRESSION) #####################
    trainData = pd.read_csv(os.path.join(os.getcwd(), "Training_Dataset.csv"))  # Re-Load in Training Dataset
    priceData = cleanData(trainData, "Training", "Price")

    # Random Forest Regressor
    def rfModel2(df):
        df = df.copy()
        vehicle_y = df.pop("Dealer_Listing_Price")
        start_rf = time.time()
        print("Starting Random Forest Grid Search")
        train_x, test_x, train_y, test_y = train_test_split(df, vehicle_y, test_size=0.2, random_state=322)
        param_grid = {'n_estimators': [10, 25, 50, 100], 'max_depth' : [5, 10, 15, 20]}
        rf_model = RandomForestRegressor(random_state=322)
        grid_rf_model = GridSearchCV(rf_model, param_grid, cv=10, n_jobs=4, scoring="r2")
        grid_rf_model.fit(train_x, train_y)
        print("Done with Random Forest Grid Search")
        end_rf = time.time()
        print(time.strftime("%Hh%Mm%Ss", time.gmtime(end_rf - start_rf)))
        rf_predict = grid_rf_model.predict(test_x)
        print('Random Forest R^2: ' + str(metrics.r2_score(test_y, rf_predict)))
        print("Best Parameters: " + str(grid_rf_model.best_params_))
        return grid_rf_model

    # XGBoost Regressor
    def xgbModel2(df) :
        df = df.copy()
        vehicle_y = df.pop("Dealer_Listing_Price")
        start_rf = time.time()
        print("Starting XGBoost Grid Search")
        train_x, test_x, train_y, test_y = train_test_split(df, vehicle_y, test_size=0.2, random_state=322)
        param_grid = {'n_estimators': [10, 25, 50, 100], 'max_depth' : [3, 5, 7, 9]}
        xgb_model = XGBRegressor(random_state=322)
        grid_xgb_model = GridSearchCV(xgb_model, param_grid, cv=10, n_jobs=4) #Default is R^2
        grid_xgb_model.fit(train_x, train_y)
        print("Done with XGBoost Grid Search")
        end_rf = time.time()
        print(time.strftime("%Hh%Mm%Ss", time.gmtime(end_rf - start_rf)))
        xgb_predict = grid_xgb_model.predict(test_x)
        print("XGBoost R^2: " + str(metrics.r2_score(test_y, xgb_predict)))
        print("Best Parameters: " + str(grid_xgb_model.best_params_))
        return grid_xgb_model

    # Ridge Regression with alpha hyperparameter tuning
    # Nota Bene: Could consider using log(Price) since it's a price, to test alternative specification performance.
    def rrModel(df) :
        df = df.copy()
        vehicle_y = df.pop("Dealer_Listing_Price")
        start_rf = time.time()
        print("Starting Regression")
        train_x, test_x, train_y, test_y = train_test_split(df, vehicle_y, test_size = 0.2, random_state = 322)
        reg = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], scoring = 'r2', cv = 10)
        reg.fit(train_x, train_y)
        print("Done with Regression")
        end_rf = time.time()
        print(time.strftime("%Hh%Mm%Ss", time.gmtime(end_rf - start_rf)))
        rr_predict = reg.predict(test_x)
        print('Regression R^2: ' + str(metrics.r2_score(test_y, rr_predict))) #R^2 is 0.6839
        print("Best Parameters: Alpha = " + str(reg.alpha_))
        return rr_predict

    ### Run the Models ###
    priceRF = rfModel2(priceData) #1m40seconds, 0.7799 R^2
    priceXGB = xgbModel2(priceData) #8 seconds, 0.7621 R^2
    priceRR = rrModel(priceData) #<1 seconds, 0.6839 R^2

    # Random Forest provides the best performance in terms of R^2, so we use it on Test Data.
    # Similar comments about cleaning as with the Classification Scenario.
    testData = pd.read_csv(os.path.join(os.getcwd(), "Test_Dataset.csv"))
    priceTest = cleanData(testData, "Test", "Price")
    priceID = priceTest.pop("ListingID")
    test_cols = priceTest.columns
    train_cols = priceData.columns
    set(test_cols) - set(train_cols)  # empty; all Train columns are in Test
    missing = list(set(train_cols) - set(test_cols))  # there's some train columns not in test.
    missing.remove("Dealer_Listing_Price")
    for m in missing:
        priceTest[m] = False

    colOrder = train_cols.to_list()
    colOrder.remove("Dealer_Listing_Price")

    priceTest = priceTest[colOrder]
    priceOut = pd.Series(priceRF.predict(priceTest), name="Dealer_Listing_Price")
    priceID = priceID.reset_index(drop = True)
    priceOut = priceOut.reset_index(drop = True)
    priceConcat = pd.concat([priceID, priceOut], axis = 1)
    priceOut = priceConcat.sort_values(["ListingID"], ascending = True).reset_index(drop = True) #Final Regression (Dealer_Listing_Price) Test Output

    ################### COMBINE TO GET FINAL OUTPUT AND SAVE AS .CSV ###################
    finalOut = trimOut.merge(priceOut, how = "outer", on = "ListingID", validate = "1:1")
    finalOut = finalOut.sort_values(["ListingID"], ascending = True).reset_index(drop = True)
    finalOut.to_csv("Test_Output.csv", index = False)

    print("Program Done!")