import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely import shortest_line


def read_data(filename, file_epsg):
    """
    Reads in a .gpkg filetype, sets the epsg for this data file, and keeps a set of columns for post analysis.

    Args:
        filename (string): File path to .gpkg file.
        file_epsg (int): epsg description using integer.

    Returns:
        geopandas dataframe: Filtered dataframe.
    """
        
    data = gpd.read_file(filename)
    data = data.set_crs(epsg=file_epsg)

    columns_to_keep = ['rock_type', 'unit_desc', 'geometry']

    return data[columns_to_keep]

#process data to get subset of data relavent for two rock types 
def process_data(df, rock_types_string_a, rock_types_string_b):
    """
    This function seperates the data contained in a dataframe by two given rock types as [type, subset1, subset2...].

    For example, if we are told that: Serpentinite is a subtype of ultramafic rock, i.e. all ultramafic rocks are also serpentinite
    then you would have the rock_type_string as = ["serpentinite", "ultramafic"]. This function looks for these strings in the
    df's rock_type column. Note you must know what to look for (good to see np.unique(df['rock_type])) for example the word 
    "granodiorite" may not exist so instead you have to search for "granodioritic".

    Args:
        df (geopandas dataframe): Output dataframe from read_data() .
        rock_sypes_string_a (list): A list of strings, with the first element being the 
        rock type and the subsuqent elements being any subsets of that rock type to include
        rock_sypes_string_b (list): A list of strings, with the first element being the 
        rock type and the subsuqent elements being any subsets of that rock type to include.

    Returns:
        tuple: A merged dataframe containing just the two rock types, a dataframe containing just rock type a, 
        and a dataframe containing just rock type b.

    """

    #print(np.unique(df['rock_type']))
    pattern_a = '|'.join(rock_types_string_a)
    filtered_rock_a_df = df[df['rock_type'].str.contains(pattern_a, na=False)]

    pattern_b = '|'.join(rock_types_string_b)
    filtered_rock_b_df = df[df['rock_type'].str.contains(pattern_b, na=False)]

    print("Seperated data in the following rock types: ")
    print("Type A: " + str(np.unique(filtered_rock_a_df['rock_type'])))
    print("Type B: " + str(np.unique(filtered_rock_b_df['rock_type'])))
         
    final = pd.concat([filtered_rock_a_df, filtered_rock_b_df], ignore_index=True)

    return gpd.GeoDataFrame(final, crs=df.crs), filtered_rock_a_df.reset_index(drop=True), filtered_rock_b_df.reset_index(drop=True)


def overlap_between_rock_types(df_rock_a, df_rock_b, proximity_threshold_m):
    """
    This function returns the shapes (Lines, Points etc.) that correspond to overlaps and close proximity between every combination 
    of shapes contained in df_rock_a and df_rock_b (per row).

    The method is as follows:
    1) loop over each shape in df_rock_a contained in df_rock_a['geometry]
    2) check for all cases of intersection (overlap, embedded, boundary, point) between this query shape in a and all shapes in b
       this is done through df_rock_b.intersection(query_a_polygon); if the result is a polygon or multipolygon
       (indicating area overlap), the boundary of that geometry is used so that only the interface between rock
       types is considered
    3) store these overlap geometries in the output dataframe
    4) Compute the shortest line distance between this query shape and all of the shapes in df_rock_b
    5) If any pair corresponds to a distance not equal to zero (as this would have been taken care of in the overlap case)
       and the distance is less than the proximity_threshold_m, then add the shortest line (shapely built in function) 
       shape to the output dataframe.

    Args:
        df_rock_a (geopandas dataframe): Dataframe containing shapes of a particular rock type (see output of process_data() ).
        df_rock_b (geopandas dataframe): Dataframe containing shapes of a particular rock type (see output of process_data() ).
        proximity_threshold_m (scalar): maximum distance in meters, beyond which two shapes are not considered within close proximity to each other

    Returns:
        geopandas dataframe: Contains all intersections and close proximity shapes.

    Note: 
        There is room for optimization here, however I chose this brute force method initially for easier debugging and readability. 
    """

    if df_rock_a.crs != df_rock_b.crs:
        print("ERROR: crs is not consistenet between rock type A/B")
        return gpd.GeoDataFrame()

    crs = df_rock_a.crs
    intersections = []

    for row_a in df_rock_a.itertuples():
        query_a_polygon = row_a.geometry

        #check for explicit overlap intersection (overlap, embedded, boundary, point)
        explicit_intersections_query_a = df_rock_b.intersection(query_a_polygon)
        #remove any empty rows
        explicit_intersections_query_a = explicit_intersections_query_a[~explicit_intersections_query_a.is_empty]

        #if intersection is a polygon, just store the boundary, else store the intersection 
        for geom in explicit_intersections_query_a:
            if geom.geom_type in ["Polygon", "MultiPolygon"]:
                geom = geom.boundary
            intersections.append(geom)

        #determine "close enough" candidates 
        distances = df_rock_b.distance(query_a_polygon) #min distance across two polygons

        #nearby (< max distances for valid intersection) and omitting those that intersect (resulting in distnace = 0)
        idx_of_interest = (~np.isclose(distances, 0.0)) & (distances <= proximity_threshold_m)

        #for those cases add the shortest line geometry
        for idx in distances.index[idx_of_interest]:
            polyb = df_rock_b['geometry'].loc[idx]
            line = shortest_line(query_a_polygon, polyb)

            if not line.is_empty:
                intersections.append(line)
          
    return gpd.GeoDataFrame(geometry=intersections, crs=crs)


def likelihood_model(rs, fall_of_distance_m):
    """
    This is a likelihood model that is a function of distance and a given fall of distance in meters. 
    If the distance between a point and a shape is less than the fall of distance then the likelihood value is 1. 
    If this distance exceeds the fall of distance than the value is between 0 and 1 approaching zero for large distances. 
    The value at the fall of distance is 1. 

    Args:
        rs (array or scalar): Distance in meters.
        fall_of_distance_m (scalar): Cutoff parameter, in meters, for when likelihood begins to smoothly decay.

    Returns:
        likelihood (array or scalar): likelihood value (0, 1].
    """
    rs_arr = np.asarray(rs)

    likelihood = np.where(rs_arr < fall_of_distance_m, 1.0,
        np.exp(1 - rs_arr / fall_of_distance_m))

    if np.isscalar(rs):
        return float(likelihood)

    return likelihood

def compute_likelihood_of_point(point, dataframe, fall_of_distance_m):
    """
    This utility function is used to compute the likelihood of a point given the intersection dataframe and a fall of distance in meters.
    It works by first computing the distance from the given point to all shapes in the 
    dataframe (for a shape with multiple points it uses the minimum distance from the given point and all the given points of the shape).

    The maximum likelihood is used, but can extend this to mean or other single value statistical metrics. 

    Args:
        point (Point): A shapely Point object.
        dataframe (geopandas dataframe): A dataframe containing shapes and polyogons.

    Returns:
        float: maximum likelihood value across from this point and all shapes in the dataframe.
    """

    if dataframe.is_empty.all():
        return 0.0

    distances_from_point = dataframe.distance(point)
    likelihoods = likelihood_model(distances_from_point, fall_of_distance_m)

    return np.max(likelihoods)


def compute_likelihood(full_dataset, query_test_points_EN_26910, rock_type_a, rock_type_b, fall_of_distance_m ):

    """
    This wrapper function starts from the initial dataset in the .gpkg format and returns the 
    likelihood for all the query points in query_test_points_EN_26910.

    The method is as follows: 
    1) filter data by the two rock types of interest using process_data()
    2) find overlap intersections/boundaries and close proximity regions using overlap_between_rock_types()
    3) depending on the structure of the query test points, compute the likelihood for each query test point 
        using compute_likelihood_of_point()
    
    Note: proximity_threshold_m in meters is set fixed here

    Args:
        full_dataset (geopandas dataframe): output of read_data() with shapes specifc to the easting/northing approximate local cartesian 26910 system.
        query_test_points_EN_26910 (list): query test points to compute likelihood score for. These points must be in the same coordinate frame as 
                                           the full_dataset (26910). This function can take a single point [E, N], a list of points [ [E1, N1], [E2, N2] ]
                                           or a grid of points as output from meshgrid [egrid, ngrid]
        rock_type_a (list): a list containing a type of rock and subsets (see process_data() for more information )
        rock_type_b (list): a list containing a type of rock and subsets (see process_data() for more information )
        fall_of_distance_m (scalar) : a fall of distance in meters beyond which the likelihood falls of exponentially 
                                      of finding a Cobalt deposti (see likelihood_model)

    Returns:
        tuple: likelihood (same shape as quer_test_points_EN_26910), dataframe for only rock type a rock_a_df, 
               dataframe for only rock type b rock_b_df, and a dataframe containing the intersections between the two rock types

    """

    #fixed model parameters 
    proximity_threshold_m = 1000 #m

    #filter data by rock type a/b including any subsets
    _, rock_a_df, rock_b_df = process_data(full_dataset, rock_type_a, rock_type_b)

    #explicity intersect (overlap) across all combinations in addition to close proximity neighbors
    intersections = overlap_between_rock_types(rock_a_df, rock_b_df, proximity_threshold_m)

    #Determine structure of query (single point, vector of points [N, 2], or grid [ 2, N, M ] ) 
    query_shape = np.shape(query_test_points_EN_26910)
    query_test_points_EN_26910 = np.asarray(query_test_points_EN_26910)

    #single point (2, )
    if query_shape == (2,): 
        query_point = Point( query_test_points_EN_26910[0], query_test_points_EN_26910[1] )
        likelihood = compute_likelihood_of_point(query_point, intersections, fall_of_distance_m)
        
    #set of points (N, 2)
    elif query_test_points_EN_26910.ndim == 2 and query_shape[1] == 2: 
        N = query_shape[0]
        likelihood = np.zeros(N)
        for query_point_row in range(N):
            query_point = Point( query_test_points_EN_26910[query_point_row][0], query_test_points_EN_26910[query_point_row][1] )
            likelihood[query_point_row] = compute_likelihood_of_point(query_point, intersections, fall_of_distance_m)

    #grid [2, N, M]
    elif query_test_points_EN_26910.ndim == 3 and query_shape[0] == 2: 
        Ny, Nx = query_test_points_EN_26910[0].shape
        likelihood = np.zeros([Ny, Nx])

        east_grid = query_test_points_EN_26910[0]
        north_grid = query_test_points_EN_26910[1]
        for iy in range(Ny):
            for ix in range(Nx):
                query_point = Point( east_grid[iy, ix], north_grid[iy, ix] )
                likelihood[iy, ix] = compute_likelihood_of_point(query_point, intersections, fall_of_distance_m)

    else:
        print("ERROR: input (x,y) points not in accapted structure")
        likelihood = np.zeros(np.shape(query_test_points_EN_26910))

    return likelihood, rock_a_df, rock_b_df, intersections

def run_unit_tests():

    #overlap_between_rock_types(df_rock_a, df_rock_b, proximity_threshold_m)
    points_a = [(0, 0), (1, 0), (1, 1), (0, 1)]
    points_b = [(10, 10), (11, 10), (11, 11), (10, 11)]
    dfa = gpd.GeoDataFrame(
        {"geometry": [Polygon(points_a)]}, crs=26910)
    dfb = gpd.GeoDataFrame(
        {"geometry": [Polygon(points_b)]}, crs=26910)
    intersections_present = overlap_between_rock_types(dfa, dfb, np.sqrt(2)*9.0 - 1e-10)
    if (~intersections_present['geometry'].is_empty.all()):
        print("ERROR: unit test for overlap_between_rock_types() showed non empty geometries")
        print(intersections_present)

    #compute_likelihood_of_point(point, dataframe, fall_of_distance_m) and likelihood_model(rs, fall_off_distance_m)
    likelihood_to_check = likelihood_model(10.0*np.sqrt(2), 10.0)
    likelihood_output = compute_likelihood_of_point(Point(points_a[0]), dfb, 10.0)
    if ~np.isclose( likelihood_output - likelihood_to_check, 0 ):
        print("ERROR: unit test comparing likelihood_model() and compute_likelihood_of_point() are incosistent")
    if (likelihood_to_check >= 1.0):
        print("ERROR: unit test of likelihood_model() fails requirement to fall of to zero beyond fall of distance")

    return


run_unit_tests()

#########################################################################################################
rock_type_a = ["serpentinite", "ultramafic"]
rock_type_b = ["granodioritic"]
fall_off_param = 10*(1000) #meters
input_crs = 26910

#read data file, set crs
data = read_data("Geo/BedrockP.gpkg",  input_crs)

#generate query grid over a region of interest 
Npixels = 250
eastings = np.linspace( 285000.0, 688000, Npixels)
northings = np.linspace( 5480000.0, 5935000.0, Npixels)
egrid, ngrid = np.meshgrid(eastings, northings)

#compute likelihood over easting/northing grid 
likelihoods, rock_a_df, rock_b_df, intersections = compute_likelihood(data, [egrid, ngrid] ,rock_type_a, rock_type_b, fall_off_param )

#########################################################################################################

#Plotting
fig, ax = plt.subplots(figsize=(10, 10))
data.plot(ax=ax, color="none", edgecolor="black", alpha=0.1, zorder=3)
rock_a_df.plot(ax=ax, facecolor="gold", edgecolor="gold", alpha=0.8, zorder=2)
rock_b_df.plot(ax=ax, facecolor="blue", edgecolor="blue", alpha=0.4, zorder=2)

#likelihood map
hm = ax.pcolormesh(
    egrid,
    ngrid,
    likelihoods,
    cmap="OrRd",
    shading="auto",
    alpha=0.6,
    zorder=0,
)

#intersections 
intersections.plot(
    ax=ax,
    color="green",
    linewidth=3,
    zorder=10
)
cbar = plt.colorbar(hm, ax=ax)
cbar.set_label("Cobalt Likelihood", fontsize=20)
ax.set_xlabel("Easting (m)", fontsize=20)
ax.set_ylabel("Northing (m)", fontsize=20)
ax.set_title("Rock Contacts and Cobalt Likelihood", fontsize = 20)
ax.grid()

#legend
data_patch = mpatches.Patch(color="black", label="Dataset Boundaries")
rock_a_patch = mpatches.Patch(color="gold", label="Serpentinite and Ultramafic" )
rock_b_patch = mpatches.Patch(color="blue", label="Granodiorite")
intersections_line = mlines.Line2D([], [], color="green", linewidth=3, label="Contact or Proximity")

ax.legend(handles=[data_patch, rock_a_patch, rock_b_patch, intersections_line], fontsize=20)

plt.tight_layout()
plt.show()