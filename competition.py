"""
Author: Jiachen Wu, USC-ID: 8656902544

Method Description:
This recommendation system is based on mixed collaborative filters. 

Error Distribution:
>=0 and <1: 101297
>=1 and <2: 33338
>=2 and <3: 6402
>=3 and <4: 897
>=4: 110

RMSE:
0.9904102504035428

Execution Time:

"""
import os
import csv
import sys
import json
import math
import time
from pyspark import SparkContext


user_list, business_list = list(), list()
user_table, business_table = dict(), dict()
user_dict, business_dict = dict(), dict()
user_norm, business_norm = dict(), dict()
user_sum, business_sum = dict(), dict()
business_vector, user_vector = dict(), dict()
pair_weights = dict()

dataset_dir = sys.argv[1]
train_file = os.path.join(dataset_dir, 'yelp_train.csv')
feature_file = os.path.join(dataset_dir, 'business.json')
photo_file = os.path.join(dataset_dir, 'photo.json')
user_file = os.path.join(dataset_dir, 'user.json')
test_file = sys.argv[2]
output_file = sys.argv[3]
DEBUG = True

def get_user_id(user):
    if not user in user_table:
        user_list.append(user)
        user_table[user] = len(user_list)-1
    return user_table[user]

def get_business_id(business):
    if not business in business_table:
        business_list.append(business)
        business_table[business] = len(business_list)-1
    return business_table[business]

def getPairs(ids):
    ids.sort()
    l = len(ids)
    ret = list()
    for i in range(l-1):
        for j in range(i+1, l):
            ret.append((ids[i], ids[j]))
    return ret

def pearsonCorrelation(pair):
    user1, user2 = pair[0], pair[1]
    if not user1 in user_dict or not user2 in user_dict:
        return 0
    rate1, rate2 = list(), list()
    for business in user_dict[user1]:
        if business in user_dict[user2]:
            rate1.append(user_dict[user1][business])
            rate2.append(user_dict[user2][business])
    if len(rate1) <= 50:
        return 0
    avg1 = user_sum[user1]/len(user_dict[user1])
    avg2 = user_sum[user2]/len(user_dict[user2])
    p, sum1, sum2 = 0, 0, 0
    for i in range(len(rate1)):
        p += ((rate1[i]-avg1)*(rate2[i]-avg2))
        sum1 += ((rate1[i]-avg1)**2)
        sum2 += ((rate2[i]-avg2)**2)
    weight = 0 if p == 0 else p / (sum1*sum2)**0.5
    return weight

def pearsonCorrelationItems(pair):
    bus1, bus2 = pair[0], pair[1]
    if not bus1 in business_dict or not bus2 in business_dict:
        return 0
    rate1, rate2 = list(), list()
    for user in business_dict[bus1].keys():
        if user in business_dict[bus2]:
            rate1.append(business_dict[bus1][user])
            rate2.append(business_dict[bus2][user])
    if len(rate1) <= 50:
        return 0
    avg1 = business_sum[bus1]/len(business_dict[bus1])
    avg2 = business_sum[bus2]/len(business_dict[bus2])
    p, sum1, sum2 = 0, 0.0, 0.0
    for i in range(len(rate1)):
        p += ((rate1[i]-avg1)*(rate2[i]-avg2))
        sum1 += ((rate1[i]-avg1)**2)
        sum2 += ((rate2[i]-avg2)**2)
    weight = 0 if p == 0 else p / (sum1*sum2)**0.5
    return weight

def cosineSimilarity(pair, record, norm):
    key1, key2 = pair[0], pair[1]
    if not key1 in record or not key2 in record:
        return 0
    p = 0
    for candidate in record[key1]:
        if candidate in record[key2]:
            p += (record[key1][candidate]*record[key2][candidate])
    return p / (norm[key1]*norm[key2])**0.5

def contentSimilarity(pair, source):
    if source == 'business':
        vector = business_vector
    else:
        vector = user_vector
    id1, id2 = pair[0], pair[1]
    if not id1 in vector or not id2 in vector:
        return 0
    dim = len(vector[id1])
    p, q1, q2 = 0.0, 0.0, 0.0
    for i in range(dim):
        p += vector[id1][i] * vector[id2][i]
        q1 += vector[id1][i]**2
        q2 += vector[id2][i]**2
    q = (q1*q2) ** 0.5
    return p/q

def predict(target):
    user, business = target[0], target[1]
    if user in user_dict and business in user_dict[user]:
        return (target, user_dict[user][business])
    if user in user_dict and business in business_dict:
        user_var = user_norm[user]/len(user_dict[user]) \
                   - (user_sum[user]/len(user_dict[user]))**2
        business_var = business_norm[business]/len(business_dict[business]) \
                       - (business_sum[business]/len(business_dict[business]))**2
        if business_var == 0:
            return (target, business_sum[business]/len(business_dict[business]))
        elif user_var == 0:
            return (target, user_sum[user]/len(user_dict[user]))
        user_std, business_std = user_var**0.5, business_var**0.5
        user_ratio, business_ratio = business_std/(business_std+user_std), user_std/(business_std+user_std)
        return (target, predictUserBased(target)[1]*user_ratio+predictItemBased(target)[1]*business_ratio)
        """
        if user_var < 0.92*business_var:
            return predictUserBased(target)
        else:
            return predictItemBased(target)
        """
    if user in user_dict and business in business_vector:
        return predictItemBased(target)
    if business in business_dict and user in user_vector:
        return predictUserBased(target)
    if user in user_vector:
        if business in business_vector:
            return (target, (user_vector[user][0]+business_vector[business][0])*2.5)
        else:
            return (target, user_vector[user][0]*5.0)
    elif business in business_vector:
        return (target, business_vector[business][0]*5.0)
    else:
        return (target, 3.75117)


def predictUserBased(target):
    user, business = target[0], target[1]
    if user in user_vector:
        user_avg = user_vector[user][0] * 5.0
    else:
        user_avg = user_sum[user] / len(user_dict[user])
    p, q = 0, 0
    weights = list()
    for other in business_dict[business].keys():
        pair = (user, other) if user < other else (other, user)
        if not pair in pair_weights:
            content_weight = contentSimilarity(pair, source='user')
            cos_weight = cosineSimilarity(pair, user_dict, user_norm)
            pearson_weight = pearsonCorrelation(pair)
            if pearson_weight < 0:
                if content_weight > 0:
                    pair_weights[pair] = content_weight * pearson_weight
                elif cos_weight > 0:
                    pair_weights[pair] = cos_weight * pearson_weight
                else:
                    pair_weights[pair] = pearson_weight
            elif content_weight > 0:
                pair_weights[pair] = content_weight
            elif cos_weight > 0:
                pair_weights[pair] = cos_weight
            else:
                pair_weights[pair] = pearson_weight
        weights.append((pair_weights[pair], other))
    #weights.sort(key=lambda x: -x[0])
    for i in range(len(weights)):
        other = weights[i][1]
        pair = (user, other) if user < other else (other, user)
        other_avg = user_sum[other] / len(user_dict[other])
        p += ((user_dict[other][business]-other_avg)*pair_weights[pair])
        q += abs(pair_weights[pair])
    if q == 0:
        prediction = user_avg
    else:
        prediction = user_avg+p/q
    prediction += 0.002
    prediction = max(1.0, prediction)
    prediction = min(5.0, prediction)
    return (target, prediction)


def predictItemBased(target):
    user, business = target[0], target[1]
    if business in business_vector:
        business_avg = business_vector[business][0] * 5.0
    else:
        business_avg = business_sum[business] / len(business_dict[business])
    p, q = 0.0, 0.0
    weights = list()
    for other in user_dict[user].keys():
        pair = (business, other) if business < other else (other, business)
        if not pair in pair_weights:
            cos_weight = cosineSimilarity(pair, business_dict, business_norm)
            content_weight = contentSimilarity(pair, source='business')
            pearson_weight = pearsonCorrelationItems(pair)
            if pearson_weight < 0:
                if content_weight > 0:
                    pair_weights[pair] = content_weight * pearson_weight
                elif cos_weight > 0:
                    pair_weights[pair] = cos_weight * pearson_weight
                else:
                    pair_weights[pair] = pearson_weight
            elif content_weight > 0:
                pair_weights[pair] = content_weight
            elif cos_weight > 0:
                pair_weights[pair] = cos_weight
            else:
                pair_weights[pair] = pearson_weight
        weights.append((pair_weights[pair], other))
    #weights.sort(key=lambda x: -x[0])
    for i in range(len(weights)):
        other = weights[i][1]
        pair = (business, other) if business < other else (other, business)
        other_avg = business_sum[other] / len(business_dict[other])
        p += ((business_dict[other][user]-other_avg)*pair_weights[pair])
        q += abs(pair_weights[pair])
    if q == 0 or p == 0:
        prediction = business_avg
    else:
        prediction = business_avg + p/q
    prediction += 0.002
    prediction = max(1.0, prediction)
    prediction = min(5.0, prediction)
    return (target, prediction)


def getUserInfo(user):
    return (user['user_id'], (
        user['average_stars']/5.0,
        min(1.0, user['review_count']/200.0)
        #min(1.0, user['useful']/100.0),
        #min(1.0, user['funny']/100.0),
        #min(1.0, user['cool']/100.0)
        #min(1.0, user['fans']/100.0),
    ))

def getBusinessInfo(business):
    vec = [business['stars']/5.0, min(1.0, business['review_count']/200.0)]
    v_caters, v_tv, v_kid, v_dog = 0.5, 0.5, 0.5, 0.5
    if 'attributes' in business and business['attributes']:
        if 'Caters' in business['attributes']:
            v_caters = 1.0 if business['attributes']['Caters'] == 'True' else 0.0
        if 'HasTV' in business['attributes']:
            v_tv = 1.0 if business['attributes']['HasTV'] == 'True' else 0.0
        if 'GoodForKids' in business['attributes']:
            v_kid = 1.0 if business['attributes']['GoodForKids'] == 'True' else 0.0
        if 'DogsAllowed' in business['attributes']:
            v_dog = 1.0 if business['attributes']['DogsAllowed'] == 'True' else 0.0
        """
        if 'BusinessParking' in business['attributes']:
            park_dict = json.loads(business['attributes']['BusinessParking'].replace('\'', '\"'))
            if 'street' in park_dict:
                v_park = 1.0 if park_dict['street'] else 0.0
        """
    vec += [v_caters, v_tv, v_dog, v_kid, 0.0]
    return (business['business_id'], tuple(vec))


start = time.time()
sc = SparkContext(master='local[*]', appName='inf553_competition')
sc.setLogLevel('WARN')
train_data = sc.textFile(train_file, 8)
header = train_data.first()
train_data = train_data.filter(lambda x: x != header).map(lambda x: x.split(','))
business_stars = sc.textFile(feature_file, 8) \
    .map(lambda x: (json.loads(x)['business_id'], json.loads(x)['stars'])).collectAsMap()

user_list = train_data.map(lambda x: x[0]).distinct().collect()
business_list = train_data.map(lambda x: x[1]).distinct().collect()
train_avg, train_num = 0.0, 0
for index, user in enumerate(user_list):
    user_table[user] = index
for index, business in enumerate(business_list):
    business_table[business] = index
for rating in train_data.collect():
    user_id, business_id, stars = rating[0], rating[1], float(rating[2])
    train_avg += stars
    train_num += 1
    if user_id in user_dict:
        if not business_id in user_dict[user_id]:
            user_dict[user_id][business_id] = stars
            user_norm[user_id] += stars**2
            user_sum[user_id] += stars
    else:
        user_dict[user_id] = {business_id: stars}
        user_norm[user_id] = stars**2
        user_sum[user_id] = stars
    if business_id in business_dict:
        if not user_id in business_dict[business_id]:
            business_dict[business_id][user_id] = stars
            business_norm[business_id] += stars**2
            business_sum[business_id] += stars
    else:
        business_dict[business_id] = {user_id: stars}
        business_norm[business_id] = stars**2
        business_sum[business_id] = stars
train_avg /= train_num

test_data = sc.textFile(test_file, 8)
test_header = test_data.first()
test_data = test_data.filter(lambda x: x != test_header).map(lambda x: x.split(','))

photo_count = sc.textFile(photo_file, 8).map(lambda x: (json.loads(x)['business_id'], 1)) \
    .reduceByKey(lambda x,y: x+y).collectAsMap()
#max_review_nums = sc.textFile(user_file, 8).map(lambda x: json.loads(x)['review_count']).max()
user_vector = sc.textFile(user_file, 8).map(lambda x: json.loads(x)) \
    .map(getUserInfo).collectAsMap()
#max_review_nums = sc.textFile(feature_file, 8).map(lambda x: json.loads(x)['review_count']).max()
business_vector = sc.textFile(feature_file, 8).map(lambda x: json.loads(x)) \
    .map(getBusinessInfo).collectAsMap()
for business_id in photo_count.keys():
    if business_id in business_vector:
        vec = list(business_vector[business_id])
        vec[-1] = min(1.0, photo_count[business_id]/100.0)
        business_vector[business_id] = tuple(vec)
predictions = test_data.map(lambda x: (x[0], x[1])).map(predict)

if DEBUG:
    ratesAndPreds = test_data.map(lambda x: ((x[0], x[1]), float(x[2]))) \
        .join(predictions)
    RMSE = math.sqrt(ratesAndPreds.map(lambda x: (x[1][0] - x[1][1])**2).mean())
    distribution = {}
    #diff_sum, diff_count = 0, 0
    for x in ratesAndPreds.collect():
        #diff_count += 1
        # true - predict
        #diff_sum += (x[1][0] - x[1][1])
        diff = int(abs(x[1][0] - x[1][1]))
        if diff in distribution:
            distribution[diff] += 1
        else:
            distribution[diff] = 1
    #print(diff_sum/diff_count)
    print('Error Distribution:')
    print('>=0 and <1: {}'.format(distribution[0]))
    print('>=1 and <2: {}'.format(distribution[1]))
    print('>=2 and <3: {}'.format(distribution[2]))
    print('>=3 and <4: {}'.format(distribution[3]))
    print('>=4: {}'.format(distribution[4]))
    print("Root Mean Squared Error = " + str(RMSE))

fout = open(output_file, mode='w')
fwriter = csv.writer(fout, delimiter=',', quoting=csv.QUOTE_MINIMAL)
fwriter.writerow(['user_id', ' business_id', ' prediction'])
for pair in predictions.collect():
    fwriter.writerow([str(pair[0][0]), str(pair[0][1]), pair[1]])
fout.close()

if DEBUG:
    print("Duration: ", time.time()-start)