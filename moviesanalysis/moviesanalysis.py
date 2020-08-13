### Written by Lei Xie
### 14/04/2020

import pandas as pd
import sys
import json
import ast
from sklearn import preprocessing
from sklearn import ensemble 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
from scipy.stats import pearsonr   ## only for Pearson Correlation Coefficient calculation
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression


def readfile(path1, path2):
    try:
        training = pd.read_csv(path1)
        test     = pd.read_csv(path2)
        return training, test
    except Exception as e:
        print(e)
        sys.exit()

### simple 1-X encoding ###
### assign 1,2,3,4....X to each different classes
def simple1X_encoder(training_df, column_name):
    column_track = {}
    simple_code = 1
    for index, rec in enumerate(training_df[column_name]):
        if rec not in column_track:
            column_track[rec] = simple_code       
            training_df.loc[index, column_name] = column_track[rec]
            simple_code +=1
        else:
            training_df.loc[index, column_name] = column_track[rec]
    return training_df

### normalize column values to range 0-1
def df_normalizer(training_df, column_name):
    x = training_df[[column_name]].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(x_scaled)
    training_df[column_name] = df_normalized

    return training_df


def df_clean(training_df):
    to_keep = ['cast', 'crew', 'budget','genres','original_language', 'production_companies','production_countries','release_date', 'runtime']
    df1 = training_df[to_keep]
    df2 = df1.copy()
    for index, rec in enumerate(df1['cast']):
        rec = ast.literal_eval(rec)
        df2.loc[index,'cast'] = rec[0]['name']   ### keep the 1st cast name, normally, the Actor or Actress
        sum_cast_id = 0
        for item in rec:
            sum_cast_id += item['id']
        df2.loc[index, 'sum_cast_id'] = sum_cast_id   ## new column with sum of all cast ids 

    for index, rec in enumerate(df1['crew']):
        rec = ast.literal_eval(rec)
        sum_crew_id = 0
        for item in rec:
            if item['job'] == 'Director':
                df2.loc[index, 'crew'] = item['name']   ### keep the Director name 
            sum_crew_id += item['id']
        df2.loc[index, 'sum_crew_id'] = sum_crew_id     ### new column with sum of all crew ids

    for index, rec in enumerate(df1['release_date']):
        df2.loc[index,'release_date'] = int(rec[0:4])   ### keep only year value

    for index, rec in enumerate(df1['genres']):
        rec = ast.literal_eval(rec)  
        df2.loc[index,'genres'] = rec[0]['name']        ### keep the first genres name
        sum_genres_id = 0
        for item in rec:
            sum_genres_id += item['id']
        df2.loc[index, 'sum_genres_id'] = sum_genres_id ### new column with sum of all genres ids  

    for index, rec in enumerate(df1['production_companies']):
        rec = ast.literal_eval(rec)  
        if rec:
            df2.loc[index,'production_companies'] = rec[0]['name']  ### keep the first production_companies
        else:
            df2.loc[index,'production_companies'] = df2.loc[0,'production_companies'] ### for empty value, put the first production_companies value
        
    for index, rec in enumerate(df1['production_countries']):
        rec = ast.literal_eval(rec)  
        if rec:
            df2.loc[index,'production_countries'] = rec[0]['iso_3166_1']   ### keep the first production_countries
        else: 
            df2.loc[index,'production_countries'] = 'US'                ### for empty value, put the most popular country 'US'

    # print(df2.head())
    return df2


def df_encoder(training_df):

#### one hot encoding + kmeans cluster ### 
#### Give up this method as the Pearson correlation would fluctuate between 0.3 to 2.0, extremely unstable

    # enc = OneHotEncoder()

    # cast = training_df[['cast']].to_numpy()
    # cast_onehot = enc.fit_transform(cast).toarray()
    # kmeans_cast = KMeans(n_clusters=10).fit_transform(cast_onehot)
    # training_df['cast'] = kmeans_cast

    # crew = training_df[['crew']].to_numpy()
    # crew_onehot = enc.fit_transform(crew).toarray()
    # kmeans_crew = KMeans(n_clusters=10).fit_transform(crew_onehot)
    # training_df['crew'] = kmeans_crew    


#### simple 1-X encoding, X is the number of total clases ####

    training_df = simple1X_encoder(training_df, 'cast')
    training_df = simple1X_encoder(training_df, 'crew')
    training_df = simple1X_encoder(training_df, 'genres')
    training_df = simple1X_encoder(training_df, 'production_companies')
    training_df = simple1X_encoder(training_df, 'production_countries')
    training_df = simple1X_encoder(training_df, 'original_language')

#### Normalize values to 0-1 ######

    training_df = df_normalizer(training_df, 'cast')
    training_df = df_normalizer(training_df, 'crew')
    training_df = df_normalizer(training_df, 'genres')
    training_df = df_normalizer(training_df, 'production_companies')
    training_df = df_normalizer(training_df, 'production_countries')
    training_df = df_normalizer(training_df, 'original_language')
    training_df = df_normalizer(training_df, 'runtime')
    training_df = df_normalizer(training_df, 'release_date')
    training_df = df_normalizer(training_df, 'sum_crew_id')
    training_df = df_normalizer(training_df, 'sum_genres_id')
    training_df = df_normalizer(training_df, 'sum_cast_id')
    # training_df = df_normalizer(training_df, 'budget')   ## if normalize budget value, the correlation value will drop a lot

    # print(training_df.head())
    return training_df
        


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: zID.py path1 path2')
        sys.exit()
    path1 = sys.argv[1]
    path2 = sys.argv[2]
  
    training_df, validate_df = readfile(path1, path2)
    y_train_revenue     = training_df['revenue']
    y_train_rating      = training_df['rating']
    cleaned_training_df = df_clean(training_df)
    cleaned_validate_df = df_clean(validate_df)
    X_train             = df_encoder(cleaned_training_df)
    X_test              = df_encoder(cleaned_validate_df)

#### PART1 output #####     
    # regr    = LinearRegression().fit(X_train, y_train_revenue)               
    # predict_revenue = regr.predict(X_test)
 
    params = {'n_estimators': 400, 'max_depth': 4, 'min_samples_split': 2,          
          'learning_rate': 0.01, 'loss': 'ls'}
    GBDT = ensemble.GradientBoostingRegressor(**params)
    GBDT.fit(X_train, y_train_revenue)
    predict_revenue = GBDT.predict(X_test)

    part1_output = pd.DataFrame()
    part1_output['movie_id'] = validate_df['movie_id']
    part1_output['predicted_revenue'] = predict_revenue.astype(int)
    part1_output.to_csv('z3457022.PART1.output.csv', index=False)

#### PART1 summary #####
    MSR = mean_squared_error(validate_df['revenue'], predict_revenue)
    Pearson_correlation = pearsonr(validate_df['revenue'], predict_revenue)
    ## return tuple (Pearson’s correlation coefficient, Two-tailed p-value)   
    correlation = Pearson_correlation[0] 
    correlation = round(correlation, 2)  ## keep only two decimal places

    data = {
            'zid': 'z3457022',    ## 'zid': ['z3457022'],
            'MSR': MSR,           ## 'MSR': [MSR],
            'correlation': correlation    ## 'correlation': [correlation]
    }
    part1_summary = pd.DataFrame(data, index=[0])  ## if value is list, no need for index
    part1_summary.to_csv('z3457022.PART1.summary.csv', index=False)
    # print(part1_summary)
#         zid           MSR  correlation
# 0  z3457022  9.762945e+15          0.3


#### PART2 output #####
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train_rating)
    predict_rating = classifier.predict(X_test)

    part2_output = pd.DataFrame()
    part2_output['movie_id'] = validate_df['movie_id']
    part2_output['predicted_rating'] = predict_rating
    part2_output.to_csv('z3457022.PART2.output.csv', index=False)

#### PART2 summary #####
    average_precision = precision_score(validate_df['rating'], predict_rating, average='macro')
    average_recall    = recall_score(validate_df['rating'], predict_rating, average='macro')
    accuracy          = accuracy_score(validate_df['rating'], predict_rating)
    average_precision = round(average_precision, 2)   ## keep only two decimal places
    average_recall    = round(average_recall, 2) 
    accuracy          = round(accuracy, 2) 

    data = {
            'zid': 'z3457022',    
            'average_precision': average_precision,          
            'average_recall': average_recall,
            'accuracy': accuracy
    }
    part2_summary = pd.DataFrame(data, index=[0]) 
    part2_summary.to_csv('z3457022.PART2.summary.csv', index=False)
#     print(part2_summary)
#         zid  average_precision  average_recall  accuracy
# 0  z3457022               0.54            0.53      0.64





        