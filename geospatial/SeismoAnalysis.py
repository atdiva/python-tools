import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas
import util as util
import os
import pywt
from scipy import signal
import math
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.rotate import rotate_ne_rt

'''
Epicenter Event: Mag 4.4 - 30 km SSW of Alamo, Nevada, 2026-04-29 15:06:13 (UTC) 37.1132°N 115.3011°W 4.0 km depth 

Page to request event data from various stations: 
https://ds.iris.edu/wilber3/find_stations/12101482

IRIS pages for this event 
1) https://ds.iris.edu/spud/event/12101482
2) https://ds.iris.edu/ds/nodes/dmc/tools/event/12101482/

Data used for NV31 (BHE, BHN, BHZ) 10 minutes before and after S wave arrival (40 samples per sec) 
https://ds.iris.edu/pub/userdata/wilber/john-smith/2026-04-29-mwr44-southern-nevada-2/timeseries_data/

Other useful links:
1) USGS earthquake catalog, searchable by event type https://earthquake.usgs.gov/earthquakes/search/
2) IRIS dataset documentation chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://epic.earthscope.org/webfm_send/2134#:~:text=F,O
3) Multi-Modal DAG-4 Dataset - Access to underground nuclear explosion data set https://data.earthscope.org/archive/assembled/23-001/ (pdf for datset chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.osti.gov/servlets/purl/1983737)
'''

def compute_stft(dataframe):
    if not dataframe['seconds'].is_monotonic_increasing:
        print("ERROR: dataframe seconds is not monotonically increasing")
    x = dataframe['Count'].values
    dt = np.median(np.diff(dataframe['seconds']))
    fs = 1.0 / dt
    f, t, Zxx = signal.stft(x, fs, nperseg=256)
    return f, t, Zxx

def plot_stft(dataframe):
    f, t, Zxx = compute_stft(dataframe)
    mag = np.abs(Zxx)
    mag_log = np.log10(mag + 1e-12)
    vmin = np.percentile(mag_log, 1)
    vmax = np.percentile(mag_log, 99)

    plt.pcolormesh(t, f, mag_log, vmin=vmin, vmax=vmax, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    return 

def plot_asd(t, x, channel):
    f, asd = util.compute_asd(t, x)
    plt.loglog(f, asd, label=channel)

def compute_horizontal(dfE, dfN, tol=1e-6):
    tE = dfE["seconds"].to_numpy()
    tN = dfN["seconds"].to_numpy()

    if len(tE) != len(tN):
        print("ERROR: E/N have different lengths")
        return pandas.DataFrame()

    if not np.allclose(tE, tN, atol=tol, rtol=0):
        print("ERROR: E/N are not time aligned")
        return pandas.DataFrame()

    df_H = dfN.copy()
    df_H["Channel"] = "BH-Horizontal"
    df_H["Count"] = np.hypot(dfN["Count"], dfE["Count"])

    return df_H

def process_counts_per_channel_and_station(
    dataframe,
    taper_alpha=0.25
):
    dataframe = dataframe.copy()

    def process_trace(x):
        x = x.to_numpy(dtype=float)

        # demean
        x = x - np.mean(x)

        # detrend
        x = signal.detrend(x, type="linear")

        # taper
        taper = signal.windows.tukey(len(x), alpha=taper_alpha)
        x = x * taper

        return x

    dataframe["Count"] = dataframe.groupby(
        ["Station", "Channel"]
    )["Count"].transform(process_trace)

    return dataframe

def separate_by_channel_ENZ(dataframe, channels):
    df_N = (
        dataframe[dataframe["Channel"] == channels["N"]]
        .sort_values("seconds")
        .reset_index(drop=True)
        .copy()
    )
    df_E = (
        dataframe[dataframe["Channel"] == channels["E"]]
        .sort_values("seconds")
        .reset_index(drop=True)
        .copy()
    )
    df_Z = (
        dataframe[dataframe["Channel"] == channels["Z"]]
        .sort_values("seconds")
        .reset_index(drop=True)
        .copy()
    )
    return df_E, df_N, df_Z

def compute_RT(dataframe, channels, geodetic_pos, station, source_key):

    data = dataframe[dataframe['Station'] == station].copy()

    dfE, dfN, _ = separate_by_channel_ENZ(data, channels)

    #see https://seismo-live.github.io/html/Rotational%20Seismology/download+preprocess_data_wrapper.html#:~:text=In%20order%20to%20align%20the,distance%20and%20the%20azimuth%20angle.
    source_latitude, source_longitude, _ = geodetic_pos[source_key]
    station_latitude, station_longitude, _ = geodetic_pos[station]

    distance_m, _, baz_deg = gps2dist_azimuth(
        source_latitude,
        source_longitude,
        station_latitude,
        station_longitude
    )

    baz_rad = np.deg2rad(baz_deg)

    if len(dfE) != len(dfN):
        print("ERROR: E and N components have different lengths")

    if not np.allclose(dfE["seconds"].to_numpy(), dfN["seconds"].to_numpy(), atol=1e-6):
        print("ERROR: E and N components are not time-aligned")

    E = dfE["Count"].to_numpy()
    N = dfN["Count"].to_numpy()

    R = N*np.cos(baz_rad) + E*np.sin(baz_rad)
    T = -N*np.sin(baz_rad) + E*np.cos(baz_rad)

    dfR = dfE.copy()
    dfT = dfE.copy()

    dfR["Channel"] = "BHR"
    dfT["Channel"] = "BHT"

    dfR["Count"] = R
    dfT["Count"] = T

    print("INFO: " + source_key + " distance from " + station + " = " + str(distance_m/1000) + " km")

    return dfR, dfT

########################################################################################################

channels_map = {"E":"BHE",
            "N":"BHN",
            "Z": "BHZ"}

geodetic_positions = {"NV31": (38.43, -118.16, 1509), 
                      "epicenter": (37.1132, -115.3011, -4000)}

#load in data
folder_path = 'data/Seismo/429A51AnalysisM4.4/'
# Get all entries and filter to keep only files
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
nets = np.unique([file[0:2] for file in files])

#loop over each net and get data
dfn = [] 
for net in nets:
    matches = [filename for filename in files if net in filename]
    for i, file in enumerate(matches):
        filename_string = file.split(".")
        if filename_string[4] == "D": #D files are raw outputs
            continue
        else:
            station_id = filename_string[1]
            net_id = filename_string[0]
            channel_id = filename_string[3]
            fulldata = pandas.read_csv(folder_path + file)

            idx = np.where(fulldata.index == "Time")
            data = fulldata[idx[0][0] + 1:].copy()
            data.loc[:, 'UTC Time'] = data.index
            data.rename(columns={data.columns[0]: 'Count'}, inplace=True)
            data.loc[:, 'Channel'] = channel_id
            data.loc[:, 'Net'] = net_id
            data.loc[:, 'Station'] = station_id
            dfn.append(data)
df = pandas.concat(dfn, ignore_index=True)
df['Count'] = [float(count) for count in df['Count']]

#Conver to seconds since 12:00 AM PDT 4-29
df["Datetime UTC"] = pandas.to_datetime(df["UTC Time"], utc=True)
df["Pacific UDT Time"] = df["Datetime UTC"].dt.tz_convert("America/Los_Angeles")
df["seconds"] = (
    df["Pacific UDT Time"] - df["Pacific UDT Time"].dt.normalize()
).dt.total_seconds()

#process (grouped by channel and station)
df = process_counts_per_channel_and_station(df)

#seperate by channels
df_E, df_N, df_Z = separate_by_channel_ENZ(df, channels_map)

#compute radial and transverse 
df_R, df_T = compute_RT(df, channels_map, geodetic_positions, "NV31", "epicenter")

########################################################################################################

plt.figure(dpi=200, figsize=(12, 10))
plt.subplot(311)
plt.plot(df_E['Datetime UTC'], df_E['Count'], linewidth=0.5)
plt.xticks(df_E.loc[:: int(len(df_E)/5), "Datetime UTC" ], fontsize=5 )
plt.xticks(rotation=10) 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.grid()
plt.ylabel("E")
plt.subplot(312)
plt.plot(df_N['Datetime UTC'], df_N['Count'], linewidth=0.5)
plt.xticks(df_N.loc[:: int(len(df_N)/5), "Datetime UTC" ], fontsize=5 )
plt.xticks(rotation=10) 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.grid()
plt.ylabel("N")
plt.subplot(313)
plt.plot(df_Z['Datetime UTC'], df_Z['Count'], linewidth=0.5)
plt.xticks(df_Z.loc[:: int(len(df_Z)/5), "Datetime UTC" ], fontsize=5 )
plt.xticks(rotation=10) 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.grid()
plt.tight_layout()
plt.ylabel("Z")

plt.figure(dpi=200, figsize=(6, 4))
plot_asd(df_N['seconds'].to_numpy(), df_N['Count'].values, channels_map["N"])
plot_asd(df_E['seconds'].to_numpy(), df_E['Count'].values, channels_map["E"])
plot_asd(df_Z['seconds'].to_numpy(), df_Z['Count'].values, channels_map["Z"])
plt.xlabel("Frequency [Hz]")
plt.ylabel("ASD [counts / sqrt(Hz)]")
plt.grid()
plt.legend()
plt.tight_layout()

plt.figure(dpi=100, figsize=(10, 8))
plt.subplot(311)
plot_stft(df_R)
plt.title("R")
plt.subplot(312)
plot_stft(df_T)
plt.title("T")
plt.subplot(313)
plot_stft(df_Z)
plt.title("Z")
plt.tight_layout()


plt.figure(dpi=200, figsize=(6, 8))
plt.subplot(211)
plt.plot(df_R['Datetime UTC'], df_R['Count'], linewidth=0.5)
plt.xticks(df_R.loc[:: int(len(df_R)/5), "Datetime UTC" ], fontsize=5 )
plt.xticks(rotation=10) 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.grid()
plt.ylabel("R")
plt.tight_layout()
plt.subplot(212)
plt.plot(df_T['Datetime UTC'], df_T['Count'], linewidth=0.5)
plt.xticks(df_T.loc[:: int(len(df_T)/5), "Datetime UTC" ], fontsize=5 )
plt.xticks(rotation=10) 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.grid()
plt.ylabel("T")
plt.tight_layout()

plt.figure(dpi=200, figsize=(6, 8))
plt.subplot(211)
plt.plot(df_R['seconds'] - np.min(df_R['seconds']), df_R['Count'], linewidth=0.5)
#plt.xticks(df_R.loc[:: int(len(df_R)/5), "seconds" ], fontsize=5 )
plt.xticks(rotation=10) 
#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.grid()
plt.ylabel("R")
plt.tight_layout()
plt.subplot(212)
plt.plot(df_T['seconds']  - np.min(df_T['seconds']), df_T['Count'], linewidth=0.5)
#plt.xticks(df_T.loc[:: int(len(df_T)/5), "seconds" ], fontsize=5 )
plt.xticks(rotation=10) 
#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.grid()
plt.ylabel("T")
plt.tight_layout()
plt.show()




