import pandas as pd
import numpy as np
import os
import snow.utils as sf
from snap.snap_qrys import qry0,qry1,qry2
import editdistance
from collections import defaultdict

import sys

from sklearn.neighbors import KDTree
from fuzzymatcher import link_table


def add_manual(snap_df):
    interactions_df = sf.from_snow(role='all_data_viewer'
                 ,db='all_data'
                 ,wh='load_wh'
                 ,q_kind='interactions')

    interactions_df = interactions_df[interactions_df.city=='Sydney']
    interactions_df = interactions_df[interactions_df.date>'2019-05-01']
    merch_bid = interactions_df[['branchId','merchant']
                               ].drop_duplicates().sort_values(by='merchant')

    snapped = snap_df.Name.unique()

    nosnap_df = merch_bid[~merch_bid.merchant.isin(snapped)].rename({'merchant':'Name'},axis=1)

    concat_df = cull_data(pd.read_csv('nomatch.csv')
              .merge(nosnap_df,on='Name',how='outer')
             ).drop_duplicates(subset=['Name','branchId']).rename({'branchId':'Id'},axis=1)

    return pd.concat([snap_df,concat_df],ignore_index=True)


def extract_coordinates(locations):
    """
    Convert array of longitude and latitude into 2 series of lat/long.
    Also turns latitude into -latitude.
    
    Input:
    ---------
    locations: numpy array of strings of lat,long.
        
    Output:
    ---------
    lat, long: two numpy series of floats.
    Errors: (int) Number of errors.
    Example:
    --------
    lat_array, long_array = extract_coordinates(google_data.geo_location)
    """
    
    Errors = 0
    lat, long = [], []
    entries = list(locations)
    for entry in entries:
        try:
            lat_str, long_str = entry.split(",")
            lat.append(-float(lat_str)) # NOTE THE -
            long.append(float(long_str))
        except:
            # Arbitrary value.
            lat.append(100000)
            long.append(100000)
            Errors += 1
            
    return np.asarray(lat).reshape((len(lat), 1)), np.asarray(long).reshape((len(long), 1)), Errors


def construct_tree(lat, long):
    """
    Construct KD-Tree. Should only be used for data with small dimensionality.
    
    Input:
    --------
    lat, long: Numpy array of latitude and longitude of merchants from Google sheets.
    
    Output:
    -------
    tree: KDTree constructed from data.
    
    Example:
    -------
    construct_tree(lat_array, long_array)
    """
    try:
        coordinates = np.concatenate((lat, long), axis=1)
        tree = KDTree(coordinates, leaf_size=2)
        return tree
    except:
        print("Error handling data.")


def query_neighbours(entry, tree, __COUNT_ERROR, __NAMES_ERROR):
    """
    Query for 5 nearest neighbours to entry in KD-tree.
    
    Input:
    --------
    entry: Numpy array of 1xd where d is dimensionality of data. For one merchant observation.
    tree: KDTree of Google sheets data to query.
    __COUNT_ERROR: The total count of errors with merchants.
    __NAMES_ERROR: List of names of merchants with error.
    
    Output:
    --------
    ind: Numpy array size 5 of indices in Google sheet dataframe that are the closest neighbours to entry.
    
    If error occurred, it will return -1 to alert the code that it did not work out.
    
    Example:
    -------
    query_neighbours(liven_merch_df.iloc[3], tree)
    
    """
    try:
        lat, long = entry.Latitude, entry.Longitude
        coordinates = np.array([lat, long]).reshape(1, 2)
        _, ind = tree.query(coordinates, k=20) # k is number of entries to return.
        return ind, __COUNT_ERROR, __NAMES_ERROR
    except:
        # ERROR.

        #__COUNT_ERROR += 1
        #__NAMES_ERROR.append((entry.name, entry))
        return -1, __COUNT_ERROR, __NAMES_ERROR


def fuzzy_match_result(entry, neighbours, __COUNT_ERROR, __NAMES_ERROR, tolerance=3):
    """
    Fuzzy match on final 5 neighbours using Levenshtein distance to get exact match.
    
    Input:
    --------
    entry: Pandas series entry of which merchant from Liven we trying to match.
    neighbours: Entries of neighbours we are trying to fuzzy match on.
    
    Output:
    --------
    result: Entry array of joined entry and closest neighbour.
    __COUNT_ERROR: Track of error.
    __NAMES_ERROR: Names of merchants.
    """
    try:
        # Turn series/array into dataframe to make it easier to join.
        temp_df = pd.DataFrame(np.array(entry).reshape(1, 6), columns=["name", "Suburb", "Lat", "Long", "City", "Id"])
        # Fuzzy matching join.

        matches_df = link_table(temp_df, neighbours, ["name", "Suburb", "City", "Id"], ["name", "rating", "num_ratings", "cuisines", "price_two", "contact", "url", "branches url", "num branches", "facility", "timetable"])
        
        if matches_df.match_score[0] is None: 
            # When score is pretty low, so do not match.
            __COUNT_ERROR += 1
            __NAMES_ERROR.append((temp_df.name, temp_df.Suburb, "Fuzzy Error - NO match"))
            return -1, __COUNT_ERROR, __NAMES_ERROR
        
        if matches_df.match_score[0] > 0.5:
            __COUNT_ERROR += 1
            __NAMES_ERROR.append((temp_df.name, temp_df.Suburb, "Fuzzy Error - BAD match"))
            return -1, __COUNT_ERROR, __NAMES_ERROR
        
        # Select columns from join.
        matches_df = matches_df[["name_left", "Suburb", "Id", "name_right", "url"]]
        # Rename columns.

        matches_df.columns =["Name", "Suburb", "Id", "Zomato Name", "url"]
        if len(matches_df)>tolerance:
            raise Exception("Issue with matching.")
        return matches_df, __COUNT_ERROR, __NAMES_ERROR
    
    except:
        return -1, __COUNT_ERROR, __NAMES_ERROR


def get_data(zomato, liven_merch_df, city, tolerance):
    """
    Query and match data.
    
    Return:
    final_df: Dataset merged.
    __COUNT_ERROR: The count of number of errors made.
    __NAMES_ERROR: The list of merchants and branches with errors.
    len(liven_merch_df): Size of original data.
    """
    __COUNT_ERROR = 0
    __NAMES_ERROR = []
    # Convert coordinates.
    lat_array, long_array, __COUNT_ERROR = extract_coordinates(zomato.geo_location)
    print("We had {} errors with extracting coordinates.\n".format(__COUNT_ERROR))
    # Need to ensure every latitude coordinate is negative.
    
    # KD Tree.
    tree = construct_tree(lat_array, long_array) 
    
    # Iterate through Sydney data and match.
    liven_merch_df = liven_merch_df[liven_merch_df.City == city.capitalize()]

    # From fuzzy match method for final result.
    final_df = pd.DataFrame(columns=["Name", "Suburb", "Id", "Zomato Name", "url"])

    for _, liven_merch_entry in liven_merch_df.iterrows():
        # Get nearest neighbours indexes.
        index_results, __COUNT_ERROR, __NAMES_ERROR = query_neighbours(liven_merch_entry, tree, __COUNT_ERROR, __NAMES_ERROR)
        if index_results is -1:
            continue
        
        # Get actual nearest neighbours.
        nearest_neighs = zomato.iloc[index_results[0]]
        
        # Fuzzy match data.
        resulting_df, __COUNT_ERROR, __NAMES_ERROR = fuzzy_match_result(liven_merch_entry, nearest_neighs, __COUNT_ERROR, __NAMES_ERROR, tolerance)
        if type(resulting_df) == int:
            continue
        final_df = final_df.append(resulting_df)
    
    return final_df, __COUNT_ERROR, __NAMES_ERROR, len(liven_merch_df), tree


def get_levenshtein(df):
    editdists = []
    for _,r in df.iterrows():
        name = r['Name']
        zomato = r['Zomato Name']
        editdists += [editdistance.eval(name,zomato)]
    df['levenshtein'] = editdists
    return df


def cull_data(all_data):
    """
    Cull redundant data.
    """
    cull = ['Uber','Secure Parking Melbourne','Taste Of Melbourne','UberEATS Sydney','UberEats','Liven Bistro','Liven Kitchen','UberEATS'
           , 'Cardly - You Write, We Post', 'Good Food & Wine Show', 'Sayers Sister [Kounta Demo]','Liven Merchandise Store','Urban Life Photography'
           , 'Airtasker','Peninsula Hot Springs','Melbourne River Cruises','Grilla House','Margaret River Gourmet Escape','Madeira','Royale Fusion'
            ,'The Bar.Ber','Tandoori Point','The Italian Cucina','Tokyo Bar','The Trust', 'Willow Golf','Januwland Theme Park'
            ,"Orita's",'Pelican ','Olive Garden','Golftec','Lash Labs','Cafe on Bourke','Ibuki House','Mint Leaf',"Tina's Noodle Kitchen"
           ,'Uber Sydney','Cardly - You Write, We Post','RSPCA','Secure Parking Sydney','UberEATS Sydney','Liven Bistro','Good Food & Wine Show','Sayers Sister [Kounta Demo]','Airtasker','Taste Of Sydney']

    return all_data[~all_data.Name.isin(cull)]


def prepare_data(city):
    zomato = pd.read_csv('data/sydney.csv'.format(city.lower()))
    branch_transactions = pd.DataFrame(sf.from_snow
                                            (role='all_data_viewer'
                                            ,db='all_data'
                                            ,wh='load_wh'
                                            ,query=qry0
                                            ,to_df=False)
                                            ,columns=['transId','branchId','city','name','branch'])

    city_branches = branch_transactions[branch_transactions.city==city].branchId.unique()
    
    glue_data = sf.from_snow(schema='glue',query='select * from {}_'.format(city),to_df=False)
    glue_df = pd.DataFrame(glue_data,columns=['Id','url'])
    
    snapnmatch = sf.from_snow(role='all_data_viewer',wh='load_wh',db='all_data',query=qry1,to_df=False)

    liven_merch_df = pd.DataFrame(snapnmatch, columns=["Name", "Branch", "Latitude", "Longitude", "City", "Id"])
    liven_merch_df = liven_merch_df.merge(glue_df,how='outer')
    liven_merch_df = liven_merch_df[liven_merch_df.url.isnull()].drop('url',axis=1)
    liven_merch_df = liven_merch_df[liven_merch_df.City==city.capitalize()]
    liven_merch_df = liven_merch_df.drop(liven_merch_df.index[0])

    liven_merch_df = cull_data(liven_merch_df)
    liven_merch_df.Id = liven_merch_df.Id.apply(float)
    liven_merch_df = liven_merch_df[liven_merch_df.Id.isin(liven_merch_df.Id.unique())] 

    actives = sf.from_snow(role='all_data_viewer',db='all_data',wh='load_wh',query=qry2,to_df=False)

    active_branches = pd.DataFrame(actives, columns=['branchId','city'])
    active_branches = active_branches[active_branches.city==city.capitalize()]

    liven_merch_df = liven_merch_df[liven_merch_df.Id.isin(active_branches.branchId.unique())]

    return zomato, liven_merch_df


def human_in_loop_sanity_check(snap_df):
    snap_df['n'] = snap_df.groupby('Name').Id.transform('count')
    success_df = snap_df[snap_df.levenshtein==0]
    matches = defaultdict(int)

    new_df = []
    for _,r in success_df.iterrows():
        matches[r['Name']] = r
        new_df += [r.to_dict()]

    for _,r in snap_df[snap_df.levenshtein>1].iterrows():
        if type(matches[r['Name']])==int:
            text = input("Is\n{}\n{} a match?\nRespond with enter, or n for false-positives. If you make an error, type 'restart'.\n\n".format(r['Name'],r['Zomato Name']))
            
            if text.lower()!='n' and text.lower()!='restart': 
                new_df += [r.to_dict()]
                matches[r['Name']] = r.to_dict()
            
            elif text.lower()=='restart':
                success_df = human_in_loop_sanity_check(snap_df)
                break
            
            else: pass

        else: new_df += [matches[r['Name']]]

    return pd.DataFrame.from_dict(new_df).drop_duplicates(subset='Id')

def populate_empty_branches(snap_df):
    interactions_df = sf.from_snow(role='all_data_viewer'
             ,db='all_data'
             ,wh='load_wh'
             ,q_kind='interactions')

    interactions_df = interactions_df[interactions_df.city=='Sydney']
    merch_bid = interactions_df[['branchId','merchant']
                            ].drop_duplicates().sort_values(by='merchant')

    snapped = snap_df.Name.unique()

    merch_bid = interactions_df[['branchId','merchant']
                           ].drop_duplicates().sort_values(by='merchant')

    unsnapped = merch_bid[merch_bid.merchant.isin(snapped)]

    urls = []
    for _,r in unsnapped[['merchant','branchId']].iterrows():
        name = r['merchant']
        if snap_df[snap_df.Id==r['branchId']].shape[0]>0:
            continue
        else:
            qry_df = snap_df[snap_df.Name==name]
            if qry_df.shape[0] > 0:
                urls += [(r['merchant']
                        ,qry_df['Suburb'].values[0]
                        ,r['branchId']
                        ,qry_df['Zomato Name'].values[0]
                        ,qry_df['url'].values[0]
                        ,qry_df['n'].values[0])]

    return pd.DataFrame(urls, columns=['Name','Suburb','Id','Zomato Name','url','n']),interactions_df


def snapmerchants(city,tolerance=3,sanity_check=False):
    zomato, liven = prepare_data(city)
    
    FINAL_DF, count_error, name_error, size_of_data, tree = get_data(zomato, liven, city, tolerance)

    try:
        nomatch = pd.read_csv('{}_nomatch.csv'.format(city.lower()))
        new_df = nomatch.merge(FINAL_DF, how='outer')
        new_df = new_df[new_df.url!='None'].drop_duplicates(subset='Name')
        FINAL_DF = new_df
    except: pass
    
    FINAL_DF.to_csv('{}_snaps.csv'.format(city))
    if sanity_check:
        FINAL_DF = human_in_loop_sanity_check(get_levenshtein(FINAL_DF))
    else:
        FINAL_DF = get_levenshtein(FINAL_DF)
    
    return FINAL_DF, count_error, name_error, size_of_data


if __name__ == '__main__':
    city = sys.argv[1].lower()
    snapmerchants(city,sanity_check=bool(sys.argv[2]))