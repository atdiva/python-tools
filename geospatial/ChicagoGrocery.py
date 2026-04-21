import numpy as np
import geopandas 
import matplotlib.pyplot as plt
from shapely import shortest_line
import geopandas
import geodatasets
from sklearn.linear_model import LinearRegression

#For each community in Chicago, find the distance to the nearest grocery store

# Read the geoda groceries dataset
df_groc = geopandas.read_file(geodatasets.get_path("geoda.groceries"))
df_comm = geopandas.read_file(geodatasets.get_path("geoda.chicago_commpop"))
df_health = geopandas.read_file(geodatasets.get_path("geoda.chicago_health")) 

df_groc = df_groc.to_crs(epsg = 26916)
df_comm = df_comm.to_crs(epsg = 26916)
df_health = df_health.to_crs(epsg = 26916)

#df_comm and df_health have polygons that are slightly off (match community, intersect)
df_data = df_health.copy()
comm_to_geom_map = df_comm.set_index('community')['geometry']

#Intersect geometries by their communities 
df_data = df_health.copy() #health has more data so I will change in place in this dataframe
for index,rowdata in df_data.iterrows():
    #Find the geometry in df_comm to this row's community 
    community = rowdata['community']
    geom_in_comm = comm_to_geom_map.get(community)
    #overwrite this geometry with the intersection of the two 
    df_data.at[index, 'geometry'] = rowdata['geometry'].intersection(geom_in_comm)

#Add feature of number of grocery stores that fall within a community 
number_of_grocery_stores_in_community = []
for jndex, rowdataj in df_data.iterrows():
    #count number of points within this community 
    bools = rowdataj['geometry'].contains(df_groc['geometry'])
    number_of_grocery_stores_in_community.append(np.sum(bools[bools == True]))
df_data['num_grocery_stores'] = number_of_grocery_stores_in_community

#bool index 1 is true, 0 is false, for wether each community contains at least one grocery store 
df_data['contains_atleast_one_store'] = [1 if num_grocery_stores > 0 else 0 for num_grocery_stores in df_data['num_grocery_stores']]

#Community centroid 
df_data['centroid_points'] = [ geom.representative_point() for geom in df_data['geometry'] ]
distances_from_centroid_to_nearest_grocery_store = []
lines = []
for kndex, rowdatak in df_data.iterrows():
    #get centroid 
    all_distances = df_groc['geometry'].distance(rowdatak['centroid_points'])
    distances_from_centroid_to_nearest_grocery_store.append(np.min(all_distances))
    line = shortest_line(df_groc['geometry'][np.argmin(all_distances)], rowdatak['centroid_points'])
    if not line.is_empty:
        lines.append(line)
df_data['dist_to_nearest_grocery_store'] = distances_from_centroid_to_nearest_grocery_store
df_data['line_to_nearest_store'] = lines
paths_to_nearest_store = geopandas.GeoDataFrame( {"geometry": df_data['line_to_nearest_store']}, crs = df_data.crs )


#Plotting
centroids = geopandas.GeoDataFrame( {"geometry": df_data['centroid_points']}, crs = df_data.crs )
fig, ax = plt.subplots(figsize=(10, 10))
df_data.plot(ax=ax, color="none", edgecolor="black", alpha=0.9, zorder=3)
df_groc.plot(ax=ax, facecolor="blue", edgecolor="blue", alpha=0.4, zorder=2)
centroids.plot(ax=ax, color="black", edgecolor="black", alpha=0.9, zorder=3)
paths_to_nearest_store.plot(
    ax=ax,
    color="green",
    linewidth=3,
    zorder=10
)
plt.grid()
plt.show()