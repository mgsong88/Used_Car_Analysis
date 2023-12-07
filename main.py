import pandas as pd
pd.set_option('mode.chained_assignment', None) #Suppresses SettingWithCopyWarning
import numpy as np
import os
import sklearn as skl
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression, RidgeCV
import time

# Assume that main.py is in same location as the Test and Train .csv files
if __name__ == '__main__':
    def cleanData(data, type, target) :
        # data = dataframe containing Training Data or Test Data
        # type = "Training" or "Test"
        # target = "Trim" or "Price"

        # First thing to do is look at our data for simple cleaning.
        # Not all of these variables are useful or easy to interpret in a model.
        # As a data scientist, we have to find a balance of interpretability of variables in our models and which to keep.
        # I notice that there are many categorical variables (e.g., Vehicle Model, Seller State, etc.)
        # I need to deal with the categorical variables in some way, as many of the variables have thousands of unique values.
        # Here is a straightforward way of dealing with some variables in the interest of this exercise:
            # Drop ListingID (just an id field, should not be useful in predicting trim or price)
            # Drop SellerListSrc (name of seller source likely doesn't matter)
            # Drop SellerCity (too granular; SellerState instead for geographical variation)
            # Drop SellerIsPriv (only about 0.2% of our data has this value = True: trainData['SellerIsPriv'].sum()/len(trainData) = 0.0022)
            # Drop SellerName (I did a quick check that each SellerName should have the same SellerRating, so these are basically redundant variables--plus I assume through domain assumption that a SellerRating somewhat accurately captures the value of the dealership's name brand)
            # Drop SellerZip (too granular; SellerState instead for geographical variation)
            # Drop VehBodystyle (quick check shows that all are SUV, so this isn't needed)
            # Replace VehColorExt with Primary Colors and then convert to dummies later
            # Replace VehColorInt with Primary Colors and then convert to dummies later
            # Replace VehEngine with 3.6, 5.7, 6.4, or "Other" (use Regex)
            # Drop VehFeats (too many to parse for this exercise; would do this in actual business setting)
            # Convert VehHistory into # of Owners, Boolean Buyback Protection Eligible, Boolean Accidents Reported.
            # Drop VehModel (quick check shows that all VehMake = Cadillac are XT5 Model, and all VehMake = Jeep are Grand Cherokee, so this is redundant)
            # Drop VehSellerNotes (not parsing text for this exercise)
            # Drop VehType (all cars are Used, no variation here)
            # Drop VehTransmission (all cars seem to be 8-Speed or provide minimal information, domain knowledge says this probably not useful then)
        # For this exercise, I went through each variable and used, e.g., trainData['VARNAME'].unique() or .nunique() to see how many unique values were in VARNAME to make the above decisions.
        # However, there are Python packages which get summary statistics of each column that are nice to use as well!

        # At least in the trainData, there are a few cars without a SellerListSrc. It turns out that a lot of information is missing in the other columns too (i.e., Engine, Fuel, etc.)
        # Therefore, I remove those rows without SellerListSrc since they are missing too much data across the row to impute. (No rows are missing SellerListSrc in Test Data, good).
        data = data[data["SellerListSrc"].notna()]
        data.drop(columns = ["SellerListSrc", "SellerCity", "SellerIsPriv", "SellerName", "SellerZip",
                             "VehBodystyle", "VehModel", "VehSellerNotes", "VehType", "VehTransmission"], inplace = True)

        # Keep the ListingID if Test Data (need to use it to merge back output)
        if type == "Training" :
            data.drop(columns = ["ListingID"], inplace = True)

        # There are also a few cars without any Mileage information. There are some rows in Test Data missing VehMileage, so replace with median for this exercise (could consider imputation methods if more time).
        data.fillna(value={"VehMileage": np.floor(data.VehMileage.astype(float).mean())}, inplace=True)

        # DROPPING SELLERLISTSRC
        # SellerListSrc: Shorten to Acronyms for ease of usage
        # def parseSeller(df) :
        #     sellerDict = {"INVENTORY COMMAND CENTER" : "ICC", "CADILLAC CERTIFIED PROGRAM" : "CCP",
        #                   "JEEP CERTIFIED PROGRAM" : "JCP", "HOMENET AUTOMOTIVE" : "HA",
        #                   "DIGITAL MOTORWORKS (DMI)" : "DMI", "MY DEALER CENTER" : "MDC",
        #                   "SELL IT YOURSELF" : "SIY", "FIVE STAR CERTIFIED PROGRAM" : "FSCP"}
        #     df["SellerListSrc"] = df["SellerListSrc"].str.upper()
        #     df.replace({"SellerListSrc" : sellerDict}, inplace = True)
        #     return df
        # data = parseSeller(data)

        # Colors: White, Black, Silver, Blue, Red, Gray, Green, Other; for both Ext and Int colors.
        # If a vehicle has multiple colors listed, ties are broken in color listing order.
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

        # Vehicle Engine:
        def parseEngine(df) :
            engine = ["3.6", "5.7", "6.4"]
            for e in engine:
                df.loc[df["VehEngine"].str.contains(e, na=False), "VehEngine"] = e
            df.loc[~df["VehEngine"].isin(engine), "VehEngine"] = "Other"
            df.fillna(value = {"VehEngine" : "Missing"}, inplace = True) #Fill NaN with "Missing" string
            return df
        data = parseEngine(data)

        # Vehicle Features:
        # Could do some text analysis but might be a little too involved for this exercise.
        # Instead, will replace VehFeats with a count of the number of features (separated by comma) in each row as a proxy.
        # I also see that most of the values in this column are 8, so fill in NaN with 8 (median and mode value).
        def parseFeatures(df) :
            df["VehFeats"] = df["VehFeats"].str.count(',') + 1 #1 Comma = 2 Items, 2 Commas = 3 Items; default 0 commas = 1 item.
            df.fillna(value = {"VehFeats" : 8}, inplace = True)
            return df
        data = parseFeatures(data)

        # Vehicle Fuel:
        # Gasoline, E84 Flex Fuel, Diesel, and Unknown. Overwhelmingly Gasoline.
        # Change to binary, isGasoline
        def parseFuel(df) :
            df["VehFuel"] = df["VehFuel"].str.upper()
            df["isGasoline"] = df["VehFuel"].str.contains("GASOLINE")
            df.insert(df.columns.get_loc("VehFuel") + 1, "isGasoline", df.pop("isGasoline"))
            df.drop(columns = "VehFuel", inplace = True)
            return df
        data = parseFuel(data)

        # Vehicle History:
        # Pull out (use Regex): # of Owners, whether Buyback Protection Eligible (Boolean), and Accident Reported (Boolean).
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

        #-------------------------------------------------------------------------------------------------------------------
        # Done parsing all the independent variables. Time to parse Vehicle_Trim.
        # Since Vehicle_Trim is the dependent variable, if missing then drop.
        # We also won't use the Selling Price since it's implied that we can only use the other 26 variables.
        # Only do this on the Test Data
        if type == "Training" :
            if target == "Trim" : #If looking for Trim, drop the Dealer Price column
                data_Trim = data.dropna(subset = ["Vehicle_Trim"]).copy()
                data_Trim.drop(columns="Dealer_Listing_Price", inplace=True)

                # I notice that Trims are by Vehicle Make. Cadillacs have specific Trims, and Jeeps have others. So, build two models.
                # For Cadillacs, the Trims can be grouped into: Base, Luxury, Platinum, Premium
                # For Jeeps, the Trims are: Laredo, Altitude, Limited, Overland, SRT, Summit, Trailhawk, Trackhawk, Upland, Sterling.
                #   Google tells me that the 75th Anniversary is a Laredo model: https://www.thebestchrysler.com/grand-cherokee-trim-levels-explained/
                #   Wikipedia Link says that:
                #   Upland -> Laredo
                #   Sterling -> Limited
                #   Trackhawk -> SRT
                def parseTrim(df) :
                    df["Vehicle_Trim"] = df["Vehicle_Trim"].str.upper()
                    cadillacTrims = ["PREMIUM", "PLATINUM", "LUXURY"] #"Premium Luxury" Trim existing means we need to replace "Premium" first!
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
                data_Trim = pd.get_dummies(data_Trim, columns = ["SellerState", "VehColorExt", "VehColorInt",
                                                                 "VehDriveTrain", "VehEngine",
                                                                 "VehPriceLabel"], drop_first = True)

                # Now, let's split the data into Jeeps and Cadillacs for two separate models.
                jeepTrim = data_Trim[data_Trim["VehMake"] == "Jeep"].copy()
                jeepTrim.drop(columns = "VehMake", inplace = True)
                cadillacTrim = data_Trim[data_Trim["VehMake"] == "Cadillac"].copy()
                cadillacTrim.drop(columns = "VehMake", inplace = True)
                return jeepTrim, cadillacTrim

            else :
                data_Price = data.dropna(subset = ["Dealer_Listing_Price"]).copy() #Drop NA's in Price
                data_Price.drop(columns = "Vehicle_Trim", inplace = True) #Don't use Trim
                data_Price = pd.get_dummies(data_Price, columns = ["SellerState", "VehColorExt",
                                                                   "VehColorInt", "VehDriveTrain",
                                                                   "VehEngine", "VehPriceLabel"], drop_first = True)
                data_Price["isJeep"] = data_Price["VehMake"] == "Jeep" #Convert VehMake into Boolean isJeep
                data_Price.insert(data_Price.columns.get_loc("VehMake") + 1, "isJeep", data_Price.pop("isJeep"))
                data_Price.drop(columns=["VehMake"], inplace = True)
                return data_Price

        else :
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

    trainData = pd.read_csv(os.path.join(os.getcwd(), "Training_Dataset.csv"))  # Load in Training Dataset
    jeepTrain, cadillacTrain = cleanData(trainData, "Training", "Trim")

    ################## RUN VEHICLE_TRIM CLASSIFICATION #####################

    def rfModel(df, type) :
        # df = jeepTrain or cadillacTrain
        # type = "Jeep" or "Cadillac"
        df = df.copy() #avoids changing the actual underlying dataset
        vehicle_y = df.pop("Vehicle_Trim")
        start_rf = time.time()
        print(type)
        print("Starting Random Forest Grid Search")
        train_x, test_x, train_y, test_y = skl.model_selection.train_test_split(df, vehicle_y, test_size = 0.2, random_state = 322)
        param_grid = {'n_estimators' : [500], 'max_depth' : [10, 25, 50, 75]} #RF typically gets better with more # of trees.
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

    jeepRF = rfModel(jeepTrain, "Jeep")
    cadillacRF = rfModel(cadillacTrain, "Cadillac")

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
        train_x, test_x, train_y, test_y = skl.model_selection.train_test_split(df, vehicle_y_transform, test_size = 0.2, random_state = 322)
        # param_grid = {'n_estimators': [100, 200, 300, 400, 500], 'max_depth' : [3, 5, 7, 10]}
        param_grid = {'n_estimators': [100, 150, 200], 'max_depth' : [1, 3, 5]}
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

    jeepXGB, jeepLabels = xgbModel(jeepTrain, "Jeep")
    cadillacXGB, cadillacLabels = xgbModel(cadillacTrain, "Cadillac")

    def lrModel(df, type) :
        # df = jeepTrain or cadillacTrain
        # type = "Jeep" or "Cadillac"
        df = df.copy()  # avoids changing the actual underlying dataset
        vehicle_y = df.pop("Vehicle_Trim")
        start_rf = time.time()
        print(type)
        print("Starting Multinomial Logistic Regression")
        train_x, test_x, train_y, test_y = skl.model_selection.train_test_split(df, vehicle_y, test_size = 0.2, random_state = 322)
        lr = LogisticRegression(multi_class = "multinomial", solver = "newton-cg", max_iter = 200)
        lr.fit(train_x, train_y)
        print("Done with Multinomial Logistic Regression")
        end_rf = time.time()
        print(time.strftime("%Hh%Mm%Ss", time.gmtime(end_rf - start_rf)))
        prediction = lr.predict(test_x)
        accuracy = (prediction == test_y).sum() / len(test_y)
        lr_predict_prob = lr.predict_proba(test_x)
        print("ROC AUC: " + str(metrics.roc_auc_score(test_y, lr_predict_prob, multi_class='ovr', average='weighted')))
        print("Multinomial Logistic Regression Accuracy: " + str(accuracy))
        return lr

    jeepLR = lrModel(jeepTrain, "Jeep")
    cadillacLR = lrModel(cadillacTrain, "Cadillac")

    # It turns out that XGBoost provides the best performance both in Accuracy and AUC_ROC, so we use those models to predict the test values.
    testData = pd.read_csv(os.path.join(os.getcwd(), "Test_Dataset.csv"))  # Load in Training Dataset
    jeepTest, cadillacTest = cleanData(testData, "Test", "Trim")

    jeepID = jeepTest.pop("ListingID")
    cadillacID = cadillacTest.pop("ListingID")

    # jeepTest might not have all the Dummy Variables.
    test_cols = jeepTest.columns
    train_cols = jeepTrain.columns #Jeep and Cadillac same columns so one of the two is fine.
    set(test_cols) - set(train_cols) #empty; all Train columns are in Test
    missing = list(set(train_cols) - set(test_cols)) #there's some train columns not in test.
    missing.remove("Vehicle_Trim") #Don't need Vehicle_Trim (it messes up applying the model since it doesn't see Vehicle_Trim in X train data)
    for m in missing :
        jeepTest[m] = False #with the exception of Vehicle_Trim, all are Booleans. We don't need Vehicle_Trim anyways to predict.
        cadillacTest[m] = False

    colOrder = train_cols.to_list()
    colOrder.remove("Vehicle_Trim")

    jeepTest = jeepTest[colOrder]
    cadillacTest = cadillacTest[colOrder]
    jeepOut = pd.Series(jeepXGB.predict(jeepTest), name = "Vehicle_Trim")
    cadillacOut = pd.Series(cadillacXGB.predict(cadillacTest), name = "Vehicle_Trim")

    jeepID = jeepID.reset_index(drop = True)
    jeepOut = jeepOut.reset_index(drop = True)
    cadillacID = cadillacID.reset_index(drop = True)
    cadillacOut = cadillacOut.reset_index(drop = True)

    jeepConcat = pd.concat([jeepID, jeepOut], axis = 1)
    cadillacConcat = pd.concat([cadillacID, cadillacOut], axis = 1)

    jeep_dict = dict(zip(range(len(jeepLabels)), jeepLabels))
    cadillac_dict = dict(zip(range(len(cadillacLabels)), cadillacLabels))

    jeepConcat.replace({"Vehicle_Trim" : jeep_dict}, inplace = True)
    cadillacConcat.replace({"Vehicle_Trim" : cadillac_dict}, inplace = True)

    trimOut = pd.concat([jeepConcat, cadillacConcat], ignore_index = True)
    trimOut = trimOut.sort_values(["ListingID"], ascending = True).reset_index(drop = True)

    # DONE WITH CLASSIFICATION TASK

    ################## RUN DEALER_LISTING_PRICE #####################
    trainData = pd.read_csv(os.path.join(os.getcwd(), "Training_Dataset.csv"))  # Load in Training Dataset
    priceData = cleanData(trainData, "Training", "Price")

    def rfModel2(df):
        df = df.copy()  # avoids changing the actual underlying dataset
        vehicle_y = df.pop("Dealer_Listing_Price")
        start_rf = time.time()
        print(type)
        print("Starting Random Forest Grid Search")
        train_x, test_x, train_y, test_y = skl.model_selection.train_test_split(df, vehicle_y, test_size=0.2,
                                                                                random_state=322)
        param_grid = {'n_estimators': [100, 150, 200], 'max_depth': [10, 25, 50]}  # RF typically gets better with more # of trees.
        rf_model = RandomForestRegressor(random_state=322)
        grid_rf_model = GridSearchCV(rf_model, param_grid, cv=10, n_jobs=4, scoring="r2")
        grid_rf_model.fit(train_x, train_y)
        print("Done with Random Forest Grid Search")
        end_rf = time.time()
        print(time.strftime("%Hh%Mm%Ss", time.gmtime(end_rf - start_rf))) #4 mins
        rf_predict = grid_rf_model.predict(test_x)
        print('Random Forest R^2: ' + str(metrics.r2_score(test_y, rf_predict))) #R^2 is 0.7769
        print("Best Parameters: " + str(grid_rf_model.best_params_))
        return grid_rf_model
    priceRF = rfModel2(priceData)
    def xgbModel2(df) :
        df = df.copy()  # avoids changing the actual underlying dataset
        vehicle_y = df.pop("Dealer_Listing_Price")
        start_rf = time.time()
        print(type)
        print("Starting XGBoost Grid Search")
        train_x, test_x, train_y, test_y = skl.model_selection.train_test_split(df, vehicle_y, test_size=0.2,
                                                                                random_state=322)
        # param_grid = {'n_estimators': [100, 200, 300, 400, 500], 'max_depth' : [3, 5, 7, 10]}
        param_grid = {'n_estimators': [25, 50, 100, 150], 'max_depth': [1, 3, 5, 7, 10]}
        xgb_model = XGBRegressor(random_state=322)
        grid_xgb_model = GridSearchCV(xgb_model, param_grid, cv=10, n_jobs=4) #Default is R^2
        grid_xgb_model.fit(train_x, train_y)
        print("Done with XGBoost Grid Search")
        end_rf = time.time()
        print(time.strftime("%Hh%Mm%Ss", time.gmtime(end_rf - start_rf)))
        xgb_predict = grid_xgb_model.predict(test_x)
        print("XGBoost R^2: " + str(metrics.r2_score(test_y, xgb_predict))) #R^2 is 0.7621
        print("Best Parameters: " + str(grid_xgb_model.best_params_))
        return grid_xgb_model
    priceXGB = xgbModel2(priceData)

    # Running a Linear Regression, could consider doing log transform of Dealer_Listing_Price since it's a price to test alternative.
    def rrModel(df) :
        df = df.copy()
        vehicle_y = df.pop("Dealer_Listing_Price")
        start_rf = time.time()
        print(type)
        print("Starting Regression")
        train_x, test_x, train_y, test_y = skl.model_selection.train_test_split(df, vehicle_y, test_size = 0.2, random_state = 322)
        reg = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], scoring = 'r2', cv = 10)
        reg.fit(train_x, train_y)
        print("Done with Regression")
        end_rf = time.time()
        print(time.strftime("%Hh%Mm%Ss", time.gmtime(end_rf - start_rf)))
        rr_predict = reg.predict(test_x)
        print('Regression R^2: ' + str(metrics.r2_score(test_y, rr_predict))) #R^2 is 0.6839
        print("Best Parameters: Alpha = " + str(reg.alpha_))
        return rr_predict
    priceRR = rrModel(priceData)

    # It turns out that Random Forest provides the best performance in terms of R^2, so we use it on test.
    testData = pd.read_csv(os.path.join(os.getcwd(), "Test_Dataset.csv"))  # Load in Training Dataset
    priceTest = cleanData(testData, "Test", "Price")

    priceID = priceTest.pop("ListingID")

    # priceTest might not have all the Dummy Variables.
    test_cols = priceTest.columns
    train_cols = priceData.columns
    set(test_cols) - set(train_cols)  # empty; all Train columns are in Test
    missing = list(set(train_cols) - set(test_cols))  # there's some train columns not in test.
    missing.remove("Dealer_Listing_Price")
    for m in missing:
        priceTest[m] = False  # All missing are Booleans

    colOrder = train_cols.to_list()
    colOrder.remove("Dealer_Listing_Price")

    priceTest = priceTest[colOrder]
    priceOut = pd.Series(priceRF.predict(priceTest), name="Dealer_Listing_Price")

    priceID = priceID.reset_index(drop = True)
    priceOut = priceOut.reset_index(drop = True)
    priceConcat = pd.concat([priceID, priceOut], axis = 1)
    priceOut = priceConcat.sort_values(["ListingID"], ascending = True).reset_index(drop = True)

    ################### FINAL OUTPUT ###################
    finalOut = trimOut.merge(priceOut, how = "outer", on = "ListingID", validate = "1:1")
    finalOut = finalOut.sort_values(["ListingID"], ascending = True).reset_index(drop = True)
    finalOut.to_csv("Test_Output.csv", index = False)