import pandas as pd
import numpy as np
import sys
import random, json
from scipy.sparse.linalg import svds
from flask import Flask, render_template,request,redirect,Response,url_for

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

#read in data
trip_data = pd.read_csv("trip_data.csv")
trip_data.dropna()

#getting the average ratings and number of ratings
highest_rated_data = trip_data.groupby('ItemID')['Rating'].mean().sort_values(ascending=False)
most_rated_data = trip_data.groupby('ItemID')['Rating'].count().sort_values(ascending=False)
all_data = pd.merge(trip_data, highest_rated_data, on='ItemID', how='left')
all_data = pd.merge(all_data, most_rated_data, on='ItemID', how='left')
all_data = all_data.rename(columns={"Rating":"NumRatings","Rating_y":"AvgRating","Rating_x":"Rating"})
all_data = all_data.drop(columns=['ItemTimeZone'])
#print(all_data.columns)

# using portions of code from https://beckernick.github.io/matrix-factorization-recommender/
def matrix_factorization(city_ratings):
    global best_items
    global recommendations
    user_item_rating = all_data.pivot_table(index='UserID',columns = 'ItemID',values='Rating')
    user_items = user_item_rating.fillna(0).as_matrix()
    user_mean = np.mean(user_items,axis=1)
    user_items_normal = user_items-user_mean.reshape(-1,1)
    U,sigma,Vt = svds(user_items_normal,k=50)
    sigma = np.diag(sigma)
    predicted_ratings = np.dot(np.dot(U,sigma),Vt)+user_mean.reshape(-1,1)
    predictions = pd.DataFrame(predicted_ratings,columns = user_item_rating.columns)
    predictions['UserID'] = user_item_rating.index
    user_predictions = predictions[predictions['UserID'] == user_id]
    user_predictions = user_predictions.drop(['UserID'],axis=1)
    user_predictions = user_predictions.sort_values(by=user_predictions.index[0],ascending = False,axis=1)
    recommendations = (city_ratings
                       .merge(pd.DataFrame(user_predictions).transpose(),how='left',left_on='ItemID',right_on='ItemID')
                       .rename(columns = {user_predictions.index[0]:'Predictions'})
                       .sort_values('Predictions',ascending = False))
    recommendations.drop_duplicates(subset ="ItemID",keep ='first',inplace = True)
    most_rated_items = recommendations.NumRatings.quantile([.2])
    recommendations = recommendations[recommendations['NumRatings']>most_rated_items[0.2]]
    num_ratings = []
    location = []
    prediction = []
    t_type = []
    user_avg = all_data[all_data['UserID'] == user_id]['Rating'].mean()
    for i in range(recommendations.shape[0]):
        item_avg = all_data[all_data['ItemID'] == recommendations['ItemID'].iloc[i]]
        num_value = item_avg['NumRatings'].iloc[0]
        num_ratings.append(num_value)
        loc_value = item_avg['ItemCity'].iloc[0]
        location.append(loc_value)
        item_info = recommendations[recommendations['ItemID'] == recommendations['ItemID'].iloc[i]]
        pred_value = item_info['Predictions'].iloc[0]
        prediction.append(pred_value+user_avg)
        trip_value = item_avg['TripType'].iloc[0]
        t_type.append(trip_value)
    recommendations['num_ratings'] = num_ratings
    recommendations['location'] = location
    prediction = [round(i,2) for i in prediction]
    for i in range(len(prediction)):
        if prediction[i]>5:
            prediction[i]=5.00
    recommendations['prediction'] = prediction
    recommendations['t_type'] = t_type
    best_items = np.array(recommendations[['ItemID','num_ratings','location','prediction','t_type']])
    if trip_type == "BUSINESS":
        b_value = 0.3
        c_value = 0
        fa_value = -0.3
        fr_value = -0.15
        s_value = 0.15
    elif trip_type == "COUPLES":
        b_value = 0
        c_value = 0.3
        fa_value = -0.3
        fr_value = -0.15
        s_value = 0.15
    elif trip_type == "FAMILY":
        b_value = -0.3
        c_value = 0
        fa_value = 0.3
        fr_value = 0.15
        s_value = -0.15
    elif trip_type == "FRIENDS":
        b_value = -0.3
        c_value = 0
        fa_value = 0.15
        fr_value = 0.3
        s_value = -0.15
    else:
        b_value = 0.15
        c_value = 0
        fa_value = -0.3
        fr_value = -0.15
        s_value = 0.3
    for i in range(len(best_items)):
        if best_items[i][-1] == "BUSINESS":
            best_items[i][-2] += b_value
        elif best_items[i][-1] == "COUPLES":
            best_items[i][-2] += c_value
        elif best_items[i][-1] == "FAMILY":
            best_items[i][-2] += fa_value
        elif best_items[i][-1] == "FRIENDS":
            best_items[i][-2] += fr_value
        else:
            best_items[i][-2] += s_value

    score = []
    for i in range(recommendations.shape[0]):
        score.append(best_items[i][-2])
        
    score = [round(i,2) for i in score]
    for i in range(len(score)):
            if score[i]>5:
                score[i]=5.00
    recommendations['score'] = score
    recommendations = recommendations.sort_values(["score","prediction"],ascending = False)
    best_items = np.array(recommendations[['ItemID','num_ratings','location','prediction','t_type','score']])
    return best_items

#test data
user_id = "00D673CA0747712BD29890CB31E3C58D"
is_state = True
selected = "NY"
is_city = True
city = "NEWYORK"
num = 10
trip_type = "FAMILY"

users = pd.DataFrame(all_data['UserID'])
users.drop_duplicates(keep ='first',inplace = True)
users = users['UserID']
states = pd.DataFrame(all_data['ItemState'])
states.drop_duplicates(keep ='first',inplace = True)
states = states.sort_values('ItemState')
states = states['ItemState']
all_cities = pd.DataFrame(all_data[['ItemCity','ItemState']])
all_cities.drop_duplicates(subset = 'ItemCity',keep ='first',inplace = True)
cities = pd.DataFrame(all_cities['ItemCity'])
cities = cities.sort_values('ItemCity')
cities = cities['ItemCity']
types = pd.DataFrame(all_data['TripType'])
types.drop_duplicates(keep ='first',inplace = True)
types = types['TripType']
selected = "Any"

@app.route("/")
def login():
    return render_template("index.html",users = users)

@app.route("/receiver",methods=['POST'])
def receiver():
    global user_id
    user_id = str(request.form.get('users'))
    return redirect(url_for('search'))

@app.route("/search")
def search():
    global selected
    global states
    global cities
    global types
    selected = "Any"
    cities = all_cities['ItemCity']
    cities = cities.sort_values()
    return render_template("search.html",states = states, cities = cities, types = types,selected = selected)

@app.route("/selectstate",methods=['POST'])
def selectstate():
    global selected
    global states
    global cities
    global types
    selected = str(request.form.get('states'))
    if selected != "Any":
        cities = all_cities[all_cities['ItemState'] == selected]
        cities = cities['ItemCity']
        cities = cities.sort_values()
    else:
        cities = all_cities['ItemCity']
        cities = cities.sort_values()
    return render_template("search.html",states = states, cities = cities, types = types,selected = selected)

@app.route("/selectinfo",methods=['POST'])
def selectinfo():
    global selected
    global city
    global trip_type
    global is_state
    global is_city
    city = str(request.form.get('cities'))
    trip_type = str(request.form.get('types'))
    if selected == "Any":
        is_state = False
    else:
        is_state = True
    if city == "Any":
        is_city = False
    else:
        is_city = True
    return redirect(url_for('profile'))

@app.route("/profile")
def profile():
    global user_id
    user_info = pd.DataFrame(all_data[all_data['UserID'] == user_id])
    user_info.drop_duplicates(subset = 'UserID',keep = 'first',inplace = True)
    user_state = user_info['UserState'].values.tolist()[0]
    user_time = user_info['UserTimeZone'].values.tolist()[0]

    #filter based on search preferences
    if is_state == True:
        state_ratings = pd.DataFrame(all_data[all_data['ItemState'] == selected])
    else:
        state_ratings = pd.DataFrame(all_data)
    if is_city == True:
        city_ratings = pd.DataFrame(state_ratings[state_ratings['ItemCity'] == city])
    else:
        city_ratings = pd.DataFrame(state_ratings)

    #2D RS
    best_items = matrix_factorization(city_ratings)
    #print(best_items)
    best_items = best_items[:num]

    user_ratings = pd.DataFrame(all_data[all_data['UserID'] == user_id])
    user_ratings = user_ratings[['ItemID','Rating','ItemCity','TripType']].values.tolist()
    
    return render_template("profile.html",userid = user_id, userstate = user_state, usertime = user_time,city = city,selected = selected,ttype = trip_type,recommendations = best_items, ratings = user_ratings)

@app.route("/backhome",methods=['POST'])
def back_home():
    return redirect(url_for('login'))

@app.route("/gotosearch",methods=['POST'])
def gotosearch():
    return redirect(url_for('search'))


    
if __name__ == "__main__":
    app.run()
