'''
###
# problem statement - geoparse and geocode extracted nlp answers
###

goal:
visualize extracted nlp answers

steps:
#. geoparse.
#. geocode.
#. map.

    +-----------+
    |lorem ipsum|    +--------+
    |New York   +--->+New York|
    +-----------+    +---+----+
                         |
                         v
      XXXXXX        +----+----+
     XX    XX<------+ 40.7° n |
     XX    XX       | 74.0° w |
      XXXXXX        +---------+

dataset used:
geoparsing data (CC BY 4.0) from https://www.geonames.org/, geocoding data (ODbL) from https://www.openstreetmap.org/, and extracted answers from the covid-19 open research dataset (cord-19) initially released by the white house and its coalition of leading research groups that comprised of 13,202 scientific papers broken down into 4 subsets by source type -
biorxiv_medrxiv, comm_use_subset, noncomm_use_subset and pmc_custom_license.

$ python3 /mnt/g/Users/pie/Downloads/nih/covid19/kaggle-geo.py
$ python3 -m trace --trace --ignore-dir=$(python -c 'import sys ; print(":".join(sys.path)[1:])') /mnt/g/Users/pie/Downloads/nih/covid19/kaggle-geo.py
'''

### from packages, import required modules ###
from datetime import datetime # timestamps
import os # misc os interfaces
import json # json encoder + decoder
import sys # for constants, functions and methods of py interpreter
import pprint # pretty print

import numpy as np # linear algebra - [arrays] and [[matrices]] + math functions()
import pandas as pd # data processing, csv file i/o (e.g. pd.read_csv)
import matplotlib.pyplot as pp # plotting lib.interactive plots
# %matplotlib inline # sets the backend to inline so output is displayed inline within kaggle/jupyter notebook etc
# %matplotlib notebook # sets the backend to nbagg for interactivity
import geopandas as gpd # to allow spatial operations on geom types
from urllib import request # for opening, reading and parsing urls
from geotext import GeoText # to extract places from text
from geopy.geocoders import Nominatim # to geocode using osm's nominatim geocoder
from geopy.exc import GeocoderTimedOut # to abort call if no response is received within the specified timeout
from shapely.geometry import Point, Polygon # to manipulate and analyze planar geom obj, convert to POINT(), etc
import descartes # to use geom obj as matplotlib paths and patches

### geoparse: https://www.geonames.org/ ###
def fct_geoparse(input_url):
    response = request.urlopen(input_url)
    merge_answers = response.read().decode('utf8') # test: {type(input_url)} = <class 'str'>, {len(input_url)} = #
    
    # GeoText(merge_answers) -> geotext.geotext.GeoText object. also see spacy, geograpy3
    cities = GeoText(merge_answers).cities # case-sensitive [list]
    country_mentions = GeoText(merge_answers).country_mentions # OrderedDict([('...', ...), ('...', ...)])
    geo_file = input_url.split('/')[-1]
    
    cities_file = f'{os.getcwd()}/cities.{geo_file}'
    os.makedirs(os.path.dirname(cities_file), exist_ok=True)
    with open(cities_file, 'w') as open_file:
        json.dump(cities, open_file, indent=2, separators=(',', ': '))
    
    country_mentions_file = f'{os.getcwd()}/country_mentions.{geo_file}'
    os.makedirs(os.path.dirname(country_mentions_file), exist_ok=True)
    with open(country_mentions_file, 'w') as open_file:
        json.dump(country_mentions, open_file, indent=2, separators=(',', ': '))
    
    print('*** fct_geoparse ' + str(datetime.now()) + ' ***')
    return cities, country_mentions

cities = fct_geoparse(input_url='https://raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.3.biorxiv_medrxiv.json')[0]
fct_geoparse(input_url='https://raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.3.comm_use_subset.json')
fct_geoparse(input_url='https://raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.3.noncomm_use_subset.json')
fct_geoparse(input_url='https://raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.3.pmc_custom_license.json')

### geocode: https://nominatim.openstreetmap.org/ ###
geocoder = Nominatim(timeout=2, user_agent="app-covid19") # timeout in sec
lat_lon = []
for city in cities:
    try:
        geolocation = geocoder.geocode(city)
        if geolocation:
            # print(geolocation.raw) # test: print(geolocation.latitude, geolocation.longitude)
            lat_lon.append(geolocation)
            break # test: usage restriction
    except GeocoderTimedOut as err:
        print("Error: geocoder failed on input %s with message %s"%
             (city, err))
print(lat_lon) # [Location(Waltham, Middlesex County, Massachusetts, United States of America, (42.3756401, -71.2358004, 0.0))]

'''
### conclusion: map ###
pd_df = pd.DataFrame(lat_lon, columns=['CITYNAME', 'COORDINATES'])
pd_df.head(7)
geometry = [Point(x[1], x[0]) for x in pd_df['COORDINATES']] # switch order of lat lon
geometry[:7] # shapely.geometry.point.Point

# crs
crs = {'init': 'epsg:4326'}

# convert pd df to geo df
geo_df = gpd.GeoDataFrame(pd_df, crs=crs, geometry=geometry)
geo_df.head() # geom: POINT()

# shapefile
countries_wgs84 = gpd.read_file('/mnt/c/data/polygon/Countries_WGS84/Countries_WGS84.shp') # basemap
f, ax = pp.subplots(figsize=(16, 16))
countries_wgs84.plot(ax=ax, alpha=0.5, color='grey') # alpha = transparency
geo_df['geometry'].plot(ax=ax, markersize = 50, color = 'b', marker = '^', alpha=.5)
'''
