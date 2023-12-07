import pandas as pd
import numpy as np
import os
import sklearn as skl
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time


# Assume that main.py is in same location as the Test and Train .csv files
if __name__ == '__main__':
    trainData = pd.read_csv(os.path.join(os.getcwd(), "Training_Dataset.csv")) #Load in Training Dataset

    # First thing to do is look at our data for simple cleaning.
    # Not all of these variables are useful or easy to interpret in a model.
    # As a data scientist, we have to find a balance of interpretability of variables in our models and which to keep.
    # I notice that there are many categorical variables (e.g., Vehicle Model, Seller State, etc.)
    # I need to deal with the categorical variables in some way, as many of the variables have thousands of unique values.
    # Here is a straightforward way of dealing with some variables in the interest of this exercise:
        # Drop ListingID (just an id field, should not be useful in predicting trim or price)
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

    trainData.drop(columns = ["ListingID", "SellerCity", "SellerIsPriv", "SellerName", "SellerZip", "VehBodystyle",
                              "VehModel", "VehSellerNotes", "VehType", "VehTransmission"], inplace = True)

    # At least in the trainData, there are a few cars without a SellerListSrc. It turns out that a lot of information is missing in the other columns too (i.e., Engine, Fuel, etc.)
    # Therefore, I remove those rows without SellerListSrc since they are missing too much data across the row to impute. (No rows are missing SellerListSrc in Test Data, good).
    # There are also a few cars without any Mileage information. There are some rows in Test Data missing VehMileage, so replace with median for this exercise (could consider imputation methods if more time).
    trainData = trainData[trainData["SellerListSrc"].notna()]
    trainData.fillna(value={"VehMileage": np.floor(trainData.VehMileage.astype(float).mean())}, inplace=True)

    # SellerListSrc: Shorten to Acronyms for ease of usage
    def parseSeller(df) :
        sellerDict = {"INVENTORY COMMAND CENTER" : "ICC", "CADILLAC CERTIFIED PROGRAM" : "CCP",
                      "JEEP CERTIFIED PROGRAM" : "JCP", "HOMENET AUTOMOTIVE" : "HA",
                      "DIGITAL MOTORWORKS (DMI)" : "DMI", "MY DEALER CENTER" : "MDC",
                      "SELL IT YOURSELF" : "SIY", "FIVE STAR CERTIFIED PROGRAM" : "FSCP"}
        df["SellerListSrc"] = df["SellerListSrc"].str.upper()
        df.replace({"SellerListSrc" : sellerDict}, inplace = True)
    parseSeller(trainData)

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
    parseColors(trainData)

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
    parseDrive(trainData)

    # Vehicle Engine:
    def parseEngine(df) :
        engine = ["3.6", "5.7", "6.4"]
        for e in engine:
            df.loc[df["VehEngine"].str.contains(e, na=False), "VehEngine"] = e
        df.loc[~df["VehEngine"].isin(engine), "VehEngine"] = "Other"
        df.fillna(value = {"VehEngine" : "Missing"}, inplace = True) #Fill NaN with "Missing" string
    parseEngine(trainData)

    # Vehicle Features:
    # Could do some text analysis but might be a little too involved for this exercise.
    # Instead, will replace VehFeats with a count of the number of features (separated by comma) in each row as a proxy.
    # I also see that most of the values in this column are 8, so fill in NaN with 8 (median and mode value).
    def parseFeatures(df) :
        df["VehFeats"] = df["VehFeats"].str.count(',') + 1 #1 Comma = 2 Items, 2 Commas = 3 Items; default 0 commas = 1 item.
        df.fillna(value = {"VehFeats" : 8}, inplace = True)
    parseFeatures(trainData)

    # Vehicle Fuel:
    # Gasoline, E84 Flex Fuel, Diesel, and Unknown. Overwhelmingly Gasoline.
    # Change to binary, isGasoline
    def parseFuel(df) :
        df["VehFuel"] = df["VehFuel"].str.upper()
        df["isGasoline"] = df["VehFuel"].str.contains("GASOLINE")
        df.insert(df.columns.get_loc("VehFuel") + 1, "isGasoline", df.pop("isGasoline"))
        df.drop(columns = "VehFuel", inplace = True)
    parseFuel(trainData)

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
    parseVehHistory(trainData)

    # VehiclePriceLabel:
    # Good, Great, Fair, and Unknown categories
    def parsePriceLabel(df) :
        labelList = ["Good", "Great", "Fair"]
        df["VehPriceLabel"] = df["VehPriceLabel"].str.upper()
        for l in labelList :
            df.loc[df["VehPriceLabel"].str.contains(l, na=False), "VehPriceLabel"] = l
        df.fillna(value={"VehPriceLabel" : "UNKNOWN"}, inplace = True)
    parsePriceLabel(trainData)

    #-------------------------------------------------------------------------------------------------------------------
    # Done parsing all the independent variables. Time to parse Vehicle_Trim.
    # Since Vehicle_Trim is the dependent variable, if missing then drop.
    # We also won't use the Selling Price since it's implied that we can only use the other 26 variables.
    trainData_Trim = trainData.dropna(subset = ["Vehicle_Trim"]).copy()
    trainData_Trim.drop(columns = "Dealer_Listing_Price", inplace = True)

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
        df.loc[df["Vehicle_Trim"].str.contains("FWD", na=False), "Vehicle_Trim"] = "Base"
        jeepTrims = ["LAREDO", "ALTITUDE", "LIMITED", "OVERLAND", "SRT", "SUMMIT", "TRAILHAWK"]
        for j in jeepTrims :
            df.loc[df["Vehicle_Trim"].str.contains(j, na=False), "Vehicle_Trim"] = j
        df.loc[df["Vehicle_Trim"].str.contains("ANNIVERSARY", na=False), "Vehicle_Trim"] = "LAREDO"
        df.loc[df["Vehicle_Trim"].str.contains("UPLAND", na=False), "Vehicle_Trim"] = "LAREDO"
        df.loc[df["Vehicle_Trim"].str.contains("STERLING", na=False), "Vehicle_Trim"] = "LIMITED"
        df.loc[df["Vehicle_Trim"].str.contains("TRACKHAWK", na=False), "Vehicle_Trim"] = "SRT"
    parseTrim(trainData_Trim)

    # To run our models, we need to get Dummy Variables of the Categorical Variables:
    #   SellerListSrc, SellerState, VehColorExt, VehColorInt, VehDriveTrain, VehEngine, VehPriceLabel
    trainData_Trim = pd.get_dummies(trainData_Trim, columns = ["SellerListSrc",
                                                           "SellerState", "VehColorExt", "VehColorInt",
                                                           "VehDriveTrain", "VehEngine", "VehPriceLabel"])

    # Now, let's split the data into Jeeps and Cadillacs for two separate models.
    jeepTrim = trainData_Trim[trainData_Trim["VehMake"] == "Jeep"].copy()
    jeepTrim.drop(columns = "VehMake", inplace = True)
    cadillacTrim = trainData_Trim[trainData_Trim["VehMake"] == "Cadillac"].copy()
    cadillacTrim.drop(columns = "VehMake", inplace = True)

    # Let's use a Random Forest Classifier to predict the trainData_Trim
    # Alternatively, we could also employ e.g., XGBoost, Multinomial Logistic Regression, etc.

    # Jeep Random Forest
    jeep_y = jeepTrim.pop("Vehicle_Trim")
    start_rf = time.time()
    print("Starting Jeep Random Forest Grid Search")
    train_x, test_x, train_y, test_y = skl.model_selection.train_test_split(jeepTrim, jeep_y, test_size=0.2, random_state=322)
    param_grid = {'n_estimators': [500], 'max_depth': [10, 25, 50, 75]}  # RF typically gets better with more # of trees.
    rf_model = RandomForestClassifier(random_state=322)
    grid_rf_model = GridSearchCV(rf_model, param_grid, cv = 5, n_jobs = 4, scoring = "accuracy")
    grid_rf_model.fit(train_x, train_y)
    print("Done with RF Grid Search")
    end_rf = time.time()
    print(time.strftime("%Hh%Mm%Ss", time.gmtime(end_rf - start_rf)))  # 30 minutes
    rf_predict = grid_rf_model.predict(test_x)
    accuracy = (rf_predict == test_y).sum() / len(test_y)
    print("Jeep Accuracy: " + str(accuracy)) #66.5%
    print('Jeep Best Params:', grid_rf_model.best_params_)  # 15, 200

    # Jeep XGBoost
    start_rf = time.time()
    le = LabelEncoder()
    jeep_y_transform = le.fit_transform(jeep_y)
    train_x, test_x, train_y, test_y = skl.model_selection.train_test_split(jeepTrim, jeep_y_transform, test_size=0.2,
                                                                            random_state=322)
    print("Starting Jeep XGBoost Model")

    param_grid = {'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [3, 5, 7, 10]}
    xgb_model = XGBClassifier(random_state=322)
    grid_xgb_model = GridSearchCV(xgb_model, param_grid, cv=5, n_jobs=4)
    grid_xgb_model.fit(train_x, train_y)

    prediction = grid_xgb_model.predict(test_x)
    accuracy = (prediction == test_y).sum() / len(test_y)
    print("Done with RF Grid Search")
    end_rf = time.time()
    print(time.strftime("%Hh%Mm%Ss", time.gmtime(end_rf - start_rf)))
    print("XGBoost Accuracy: " + str(accuracy))
    print('Jeep Best Params:', grid_xgb_model.best_params_)  # 15, 200

    labels = le.classes_
    #cm = confusion_matrix(test_y, prediction)
    #ConfusionMatrixDisplay.from_predictions(test_y, prediction, display_labels=labels, xticks_rotation = "vertical")

    # Jeep Multinomial Logistic Regression
    from sklearn.linear_model import LogisticRegression
    train_x, test_x, train_y, test_y = skl.model_selection.train_test_split(jeepTrim, jeep_y, test_size=0.2,
                                                                            random_state=322)
    lr = LogisticRegression(multi_class = "multinomial", solver = "newton-cg", max_iter = 200, C = 1)
    lr.fit(train_x, train_y)
    prediction = lr.predict(test_x)
    accuracy = (prediction == test_y).sum() / len(test_y)
    print("Multinomial Logistic Accuracy: " + str(accuracy)) #Only about 60%
    #ConfusionMatrixDisplay.from_predictions(test_y, prediction, display_labels=labels, xticks_rotation="vertical")

    # Cadillac Next
    cadillac_y = cadillacTrim.pop("Vehicle_Trim")
    start_rf = time.time()
    print("Starting Random Forest Grid Search")
    train_x, test_x, train_y, test_y = skl.model_selection.train_test_split(cadillacTrim, cadillac_y, test_size=0.2,
                                                                            random_state=322)
    param_grid = {'n_estimators': [500],
                  'max_depth': [10, 25, 50, 75]}  # RF typically gets better with more # of trees.
    rf_model = RandomForestClassifier(random_state=322)
    grid_rf_model = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=4, scoring="accuracy")
    grid_rf_model.fit(train_x, train_y)
    print("Done with RF Grid Search")
    end_rf = time.time()
    print(time.strftime("%Hh%Mm%Ss", time.gmtime(end_rf - start_rf)))  # 30 minutes
    rf_predict = grid_rf_model.predict(test_x)
    accuracy = (rf_predict == test_y).sum() / len(test_y)
    print("Cadillac Accuracy: " + str(accuracy))
    print('Cadillac Best Params:', grid_rf_model.best_params_)

    # Cadillac XGBoost
    start_rf = time.time()
    le = LabelEncoder()
    cadillac_y_transform = le.fit_transform(cadillac_y)
    train_x, test_x, train_y, test_y = skl.model_selection.train_test_split(cadillacTrim, cadillac_y_transform, test_size=0.2,
                                                                            random_state=322)
    print("Starting Cadillac XGBoost Model")

    param_grid = {'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [3, 5, 7, 10]}
    xgb_model = XGBClassifier(random_state=322)
    grid_xgb_model = GridSearchCV(xgb_model, param_grid, cv=5, n_jobs=4)
    grid_xgb_model.fit(train_x, train_y)

    prediction = grid_xgb_model.predict(test_x)
    accuracy = (prediction == test_y).sum() / len(test_y)
    print("Done with RF Grid Search")
    end_rf = time.time()
    print(time.strftime("%Hh%Mm%Ss", time.gmtime(end_rf - start_rf)))
    print("XGBoost Accuracy: " + str(accuracy))
    print('Cadillac Best Params:', grid_xgb_model.best_params_)  # 15, 200

    # Cadillac Multinomial Logistic Regression
    from sklearn.linear_model import LogisticRegression
    train_x, test_x, train_y, test_y = skl.model_selection.train_test_split(cadillacTrim, cadillac_y, test_size=0.2,
                                                                            random_state=322)
    lr = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=200, C=1)
    lr.fit(train_x, train_y)
    prediction = lr.predict(test_x)
    accuracy = (prediction == test_y).sum() / len(test_y)
    print("Multinomial Logistic Accuracy: " + str(accuracy))  # Only about 60%
    # ConfusionMatrixDisplay.from_predictions(test_y, prediction, display_labels=labels, xticks_rotation="vertical")

    # TODO: Make the cleaning into one big function. Maybe do the RF into one function.
    # Update code to get each model and run it on the Test Data to get output
    # Think about running other models for comparison/example, pick best.

    # Do the Vehicle Price part as well. Regression maybe as well?

    # Clean up code, leave comments and summarize into 500 words and submit by ~noon on Thurs.