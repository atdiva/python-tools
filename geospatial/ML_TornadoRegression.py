import numpy as np 
import matplotlib.pyplot as plt
import util 
import pandas as pd
import geopandas as gpd
from shapely import shortest_line
from shapely import Point
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import sklearn  
from geopy.distance import geodesic
from sklearn.metrics import PredictionErrorDisplay

calendar = {"January":1, 
            "February":2, 
            "March":3,
            "April":4,
            "May":5,
            "June":6,
            "July":7, 
            "August":8,
            "September":9,
            "October":10,
            "November":11,
            "December":12}
epsg_USA = 1324

def apply_log(dataframe, columns):
    for column in columns:
        dataframe[column] = np.log10(dataframe[column])
    return dataframe
def apply_z_transform(dataframe, columns):
    for column in columns:
        dataframe[column] = (dataframe[column] - np.nanmean(dataframe[column]))/np.nanstd(dataframe[column])
    return dataframe
def make_zeros_small(dataframe, columns):
    for column in columns:
        dataframe[column] = [ np.random.randint(1,10)/10.0 if np.isclose(val, 0.0) else val for val in dataframe[column] ]
    return dataframe
################################################################################################################

#Map data set https://www2.census.gov/geo/tiger/GENZ2018/shp/
map4269 = gpd.read_file("../data/Climate/cb_2018_us_state_20m/cb_2018_us_state_20m.shx") 

#Storm dataset: https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
#Homepage: https://www.ncei.noaa.gov/stormevents/
raw_data = pd.read_csv("../data/Climate/StormEvents_details-ftp_v1.0_d2025_c20260323.csv")


#filter by storm type 
data = raw_data[ raw_data['EVENT_TYPE'] == "Tornado" ]
#reset index
data = data.reset_index()


#loop over and create a shortest line between beg/end of tornado event
data['TORNADO_PATH'] = [ shortest_line(Point(row.BEGIN_LON, row.BEGIN_LAT), Point(row.END_LON, row.END_LAT)) for _,row in data.iterrows() ]
fig, ax = plt.subplots(figsize=(10, 10))
map4269.plot(ax=ax, color="none", edgecolor="black", alpha=0.5)
data_gpd_plotting = data.copy()
data_gpd_plotting = gpd.GeoDataFrame(data_gpd_plotting, geometry="TORNADO_PATH", crs="EPSG:4326")
data_gpd_plotting.plot(ax=ax, color="green", alpha=1)
plt.scatter(data['BEGIN_LON'], data['BEGIN_LAT'], c='g', marker='o')
plt.scatter(data['END_LON'], data['END_LAT'], c='r', marker='x')
plt.xlim([-140, -50])
plt.ylim([20, 50])
plt.grid()
#ax.get_legend().remove()

# injuries 
data['INJURIES'] = data['INJURIES_DIRECT'] + data['INJURIES_INDIRECT']
# deaths 
data['DEATHS'] = data['DEATHS_DIRECT'] + data['DEATHS_INDIRECT']
# damage/month_name to numeric
damage = []
month = []
for _,row in data.iterrows():

    #month
    month.append( calendar[row['MONTH_NAME']] )

    #this property damage
    prop_damage_string = row.DAMAGE_PROPERTY
    #this crop damage
    crop_damage_string = row.DAMAGE_CROPS

    if type(prop_damage_string) is str:
        prop_damage = 0
        if prop_damage_string[-1] == 'K':
            prop_damage = float(prop_damage_string[:-2])*(10**3)
        elif prop_damage_string[-1] == 'M':
            prop_damage = float(prop_damage_string[:-2])*(10**6)
        elif prop_damage_string[-1] == 'B':
            prop_damage = float(prop_damage_string[:-2])*(10**9)
        else:
            print("ERROR in prop damage")

    if type(crop_damage_string) is str:
        crop_damage = 0
        if crop_damage_string[-1] == 'K':
            crop_damage = float(crop_damage_string[:-2])*(10**3)
        elif prop_damage_string[-1] == 'M':
            crop_damage = float(crop_damage_string[:-2])*(10**6)
        elif crop_damage_string[-1] == 'B':
            crop_damage = float(crop_damage_string[:-2])*(10**9)
        else:
            print("ERROR in crop damage")

    if np.isnan(prop_damage) and np.isnan(crop_damage):
        damage.append(np.nan)
    elif np.isnan(prop_damage) and ~np.isnan(crop_damage):
        damage.append(crop_damage)
    elif ~np.isnan(prop_damage) and np.isnan(crop_damage):
        damage.append(prop_damage)
    else:
        damage.append(prop_damage + crop_damage)

data["DAMAGE"] = damage
data["MONTH"] = month
data['TOR_LENGTH'] = 1609.34*data['TOR_LENGTH'] #m

#Choose features + targets
features = ['TOR_LENGTH', 'TOR_WIDTH', 'BEGIN_LON', 'MONTH']
targets = ['DAMAGE'] #DEATHS,INJURIES,DAMAGE

#Remove nans and close to zero values
filtered_data = data.copy()
#omit rows where targets == 0 or is nan
indcs_to_drop = []
for i, row in filtered_data.iterrows():
    target_values_row = row[targets]
    if np.isclose( target_values_row.all() , 0.0 ) or np.isnan( target_values_row.any() ):
        indcs_to_drop.append(i)
filtered_data = filtered_data.drop(indcs_to_drop)

#scaling
filtered_data = apply_log(filtered_data, ['TOR_LENGTH', 'TOR_WIDTH', 'MONTH'] + targets)
filtered_data = apply_z_transform(filtered_data, features + targets)
#filtered_data = make_zeros_small(filtered_data, ['DAMAGE', 'DEATHS', 'INJURIES'])

#reduce to final features and targets
filtered_data = filtered_data[features + targets]
filtered_data.hist()

#split data by latitue 
data_train = filtered_data[data['BEGIN_LAT'] > 35]
data_test = filtered_data[data['BEGIN_LAT'] < 35]

X_train = data_train[features]
y_train = data_train[targets]
X_test = data_test[features]
y_test = data_test[targets]

#Train
model = LinearRegression()
model.fit(X_train, y_train)

#Test 
predictions = model.predict(X_test)
residuals = y_test - predictions

#residual plot
plt.figure()
plt.scatter(predictions, residuals)
plt.xlabel("prediction")
plt.ylabel("residual")
plt.grid()
plt.axhline(0, c='k')

print("****************************")
print("ML Statistics")
print("****************************")
print("R2 train = " + str(model.score(X_train, y_train)))
print("R2 test = " + str(model.score(X_test, y_test)))
print(f"Coefficients: {model.coef_}")

plt.tight_layout()
plt.show()
