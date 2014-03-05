# Recommendation System Based on Yelp Data
# Using both user and feature based approaches

from recsys.algorithm.factorize import SVD
import recsys.algorithm
from recsys.datamodel.data import Data
from recsys.evaluation.prediction import RMSE, MAE
from recsys.utils.svdlibc import SVDLIBC
import csv
import math
import prettytable as pt
import shutil
import os

def distance(lat1, long1, lat2, long2):
    degrees_to_radians=math.pi/180.0
    phi1 = (90.0-lat1)*degrees_to_radians
    phi2 = (90.0-lat2)*degrees_to_radians
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians

    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1-theta2)
           + math.cos(phi1)*math.cos(phi2))
    arc = math.acos(cos)
    return arc*3963.1676

def avg_dist(num, directory):
    dist=0
    for i in range(0,len(directory)):
        dist+=distance(float(directory[(num-1)][4]), float(directory[(num-1)][5]), 
                       float(directory[i][4]), float(directory[i][5]))
    return (dist/len(directory))

def load_directory():
    with open('directory.csv') as f:
        directory = [tuple(line) for line in csv.reader(f)]
    return directory
    
def print_list(directory):
    print "Here are the restaurants I know about:\n\n"
    for i in range(0,len(directory)):
        print '%d. %s' % ((i+1), directory[i][0])
        if (i>2 and i%40==0):
            temp=raw_input("Enter a restaurant number or any other key to see more restaurants...")
            try:
                temp=int(temp)
                for i in range(0,len(directory)):
                    print '%d. %s' % ((i+1), directory[i][0])
                return temp
                break
            except:
                pass
    print "\n\nThat is all the restaurants in the database"
    return 0

def get_inputs(directory):
    inputs = []
    num = print_list(directory)
    if num == 0:
        num = int(raw_input("Which restaurant do you have an opinion on? Enter the restaurant number:   "))
    print "How would you review %s?" %directory[(num-1)][0]
    review = float(raw_input("Enter a number from 1-5 stars:   "))    
    temp = (directory[(num-1)][1], review, num)
    inputs.append(temp)

    print "Enter your reviews on at least 1 more restaurant to get personalized recommendations."
    for i in range (0,1):
        num = int(raw_input("Which other restaurant do you have an opinion on? Enter the restaurant number:   "))
        print "How would you review %s?" %directory[(num-1)][0]
        review = float(raw_input("Enter a number from 1-5 stars:   "))    
        temp = (directory[(num-1)][1], review, num)
        inputs.append(temp)
    while True:
        print """
Enter another restaurant number to provide more reviews or 'q' to see the personalized recommendations"""
        num = raw_input("Restaurant number or q to move on:")
        if num=='q':
            break
        else:
            num=int(num)
            print "How would you review %s?" %directory[(num-1)][0]
            review = float(raw_input("Enter a number from 1-5 stars:   ")) 
            temp=(directory[(num-1)][1],review, num)
            inputs.append(temp)
    return inputs

def add_to_csv(inputs):
    f = open('user_data_working.csv','a')
    for i in inputs:
        temp='1,%s,%s' %(i[0],i[1])
        f.write('\n')
        f.write(temp)
    f.close()
    
def print_choice(num, directory):
    print "\n\n\n This is the restaurant you selected....."
    y = pt.PrettyTable(["Restaurant Name", "Category", "Neighborhood", "Average stars", 
        "Num Reviews", "Avg Dist to All Restaurants (mi)"])
    y.add_row([directory[(num-1)][0], directory[(num-1)][3], directory[(num-1)][2], directory[(num-1)][6], 
        directory[(num-1)][7], avg_dist(num,directory)])
    y.float_format = 0.3
    print y

def print_reviews(input, directory):
    print "\n\n\n Here are the restaurants you reviewed....."
    y = pt.PrettyTable(["Restaurant Name", "Category", "Neighborhood", "Your review", "Average stars", 
        "Num Reviews"])
    for i in range (0,len(input)):    
        num = input[i][2]
        y.add_row([directory[(num-1)][0], directory[(num-1)][3], directory[(num-1)][2], input[i][1], directory[(num-1)][6], 
            directory[(num-1)][7]])
    y.float_format = 0.3
    print y
    
def calculate_SVD_users():
    print "Thanks for input, calculating..."
    svd = SVD()
    recsys.algorithm.VERBOSE = True
    dat_file = 'user_data_working.csv'
    svd.load_data(filename=dat_file, sep=',', 
                format = {'col':0, 'row':1, 'value': 2, 'ids': int})
    svd.compute(k=100, min_values=2, pre_normalize=None, 
                mean_center=True, post_normalize=True)
    shutil.copy('user_data_original.csv','user_data_working.csv')
    return svd

def calculate_stats_users(pct_train):
    dat_file = 'user_data_working.csv'
    data = Data()
    data.load(dat_file, sep=',', format={'col':0, 'row':1, 'value':2,'ids':int})
    train, test = data.split_train_test(percent=pct_train)               
    svd = SVD()
    svd.set_data(train)
    svd.compute(k=100, min_values=2, pre_normalize=None, mean_center=True,
    post_normalize=False)
    rmse = RMSE()
    mae = MAE()
    for rating, item_id, user_id in test.get():      
        try:
            pred_rating = svd.predict(item_id, user_id)
            rmse.add(rating, pred_rating)
            mae.add(rating, pred_rating)
        except KeyError:
            continue

    print 'RMSE=%s' % rmse.compute()
    print 'MAE=%s\n' % mae.compute()
    
def calculate_stats_features(pct_train):
    dat_file='feature_matrix.csv'
    data = Data()
    data.load(dat_file, sep=',', format={'col':0, 'row':1, 'value':2,'ids':int})
    train, test = data.split_train_test(percent=pct_train)               
    K=100
    svd = SVD()
    svd.set_data(train)
    svd.compute(k=K, min_values=0, pre_normalize=None, mean_center=False,
    post_normalize=False)
    return svd,train,test

def test_SVD(svd,train,test,pct_train):
    rmse = RMSE()
    mae = MAE()
    for rating, item_id, user_id in test.get():      
        try:
            pred_rating = svd.predict(item_id, user_id)
            rmse.add(rating, pred_rating)
            mae.add(rating, pred_rating)
        except KeyError:
            continue

    print 'RMSE=%s' % rmse.compute()
    print 'MAE=%s\n' % mae.compute()
    
def calculate_SVD_features():
    print "Thanks for input, calculating..."
    svd = SVD()
    recsys.algorithm.VERBOSE = True
    dat_file = 'feature_matrix.csv'
    svd.load_data(filename=dat_file, sep=',', 
                format = {'col':0, 'row':1, 'value': 2, 'ids': int})
    svd.compute(k=100, min_values=0, pre_normalize=None, 
                mean_center=False, post_normalize=True)
    return svd       
    
def find_recs(Uid, svd):
    recs = svd.recommend(Uid, only_unknowns=True, is_row=False)
    return recs
    
def find_similars(ID, svd):
    sim_rest = svd.similar(ID, n=10)    
    return sim_rest

def print_results(list, directory):
    print "\n\n Based on your user preferences, here are restaurants we recommend for you:"

    x = pt.PrettyTable(["Restaurant Name", "Category", "Neighborhood", "Average Stars", "Num Reviews", "Your predicted stars"])
    for value in list:
        ind=str(value[0])
        features = [v for v in directory if v[1]==ind]
        x.add_row([features[0][0], features[0][3], features[0][2], features[0][6], features[0][7], value[1]])

    x.float_format = 0.3
    print x
     
def print_results_distances(num, list, directory):
    print "\n\n Here are similar restaurants, and their match percentages:"

    x = pt.PrettyTable(["Restaurant Name", "Category", "Neighborhood", "Average Stars", "Num Reviews", "Distance (mi)", "Match rating"])
    for i,value in enumerate(list):
        if i!=0:
            ind=str(value[0])
            features = [v for v in directory if v[1]==ind]
            dist = distance (float(features[0][4]), float(features[0][5]), float(directory[(num-1)][4]), float(directory[(num-1)][5]))
            x.add_row([features[0][0], features[0][3], features[0][2], features[0][6], features[0][7], dist, value[1]])

    x.float_format = 0.3
    print x

os.system('cls' if os.name == 'nt' else 'clear')
print "\n\n"
z=pt.PrettyTable(["WELCOME TO THE BERKELEY RESTAURANT RECOMMENDATION SYSTEM"])
print z
print"\n\nHow would you like to explore restaurants?"
print"\n\n** Based on User Reviews: **\n"
print "1. Find restaurants with similar user ratings"
print "2. Build a customized recommendation just for you (Coolest option!)"
print "\n** Based on Restaurant Features: **\n"
print "3. Find restaurants with similar characteristics \n\n"
print "** Explore statistical accuracy **"
print "4. Explore statistical error rate in user approach"
print "5. Explore statistical error rate in feature approach\n\n"

choice = int(raw_input("What would you like to do?   "))

if (choice==1):
    directory = load_directory()
    num = print_list(directory)   
    if num == 0:
        num = int(raw_input("Which one do you like? Enter the number:   "))
    print_choice(num, directory)
    svd=calculate_SVD_users()    
    ID = directory[(num-1)][1]
    list = find_similars(ID, svd)
    print_results_distances(num, list, directory)
    
elif (choice==2):
    directory = load_directory()
    inputs = get_inputs(directory)
    add_to_csv(inputs)
    print_reviews(inputs, directory)
    svd = calculate_SVD_users()
    recs = find_recs(1, svd)   #The user input is assigned to user ID = '1'
    print_results(recs, directory)
    
elif (choice==3):
    directory = load_directory()
    num = print_list(directory)   
    if num == 0:
        num = int(raw_input("Which one do you like? Enter the number   "))
    print_choice(num, directory)
    svd=calculate_SVD_features()    
    ID = directory[(num-1)][1]
    list = find_similars(ID, svd)
    print_results_distances(num, list, directory)

elif (choice==4):
    pct_train = int(raw_input("What percentage for train set?   "))
    calculate_stats_users(pct_train)
    #test_SVD(svd,train,test,pct_train)

elif (choice==5):
    pct_train = int(raw_input("What percentage for train set?   "))
    svd2,train2,test2 = calculate_stats_features(pct_train)
    test_SVD(svd2,train2,test2,pct_train)

else:
    print "Sorry, I did not understand your input."
