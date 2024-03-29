#!/usr/bin/env python
# coding: utf-8

# In[53]:


import boto3
from io import BytesIO
from PIL import Image
import requests
import json
import pandas as pd
import random
import ast
import statistics
 
import ultralytics
ultralytics.checks()
from ultralytics import YOLO

#!pip install mysql-connector-python
import mysql.connector
#!pip install sqlalchemy
from sqlalchemy import create_engine

#BID = "B456"
#FID = "123"

BID = ""
FID = ""
BATABASE_API_BASE_URL = "http://ec2-52-65-94-246.ap-southeast-2.compute.amazonaws.com:5000"
TOBE_PREDICT_IMAGE_PATH = f"./images/need_to_predict_{BID}_{FID}.png"
RESULT_IMG_PATH =  f"./runs/detect/predict/need_to_predict_{BID}_{FID}.png"
MODEL = YOLO("./model//best.pt")
CONFIDENCE_LEVEL = 0.5
MEAN_FRAMES_PER_HIVE = 8

ACCESS_KEY_ID = ""
SECRET_ACCESS_KEY_ID = ""
BUCKET_NAME = "broodbox-thermal-images"

MYSQL_CREDENTIALS_MAIN = {"host":"127.0.0.1", "user":"dilshan", "password":"1234", "database":"broodbox", "port":3306}
MYSQL_CREDENTIALS_RESULTS = {"host":"127.0.0.1", "user":"dilshan", "password":"1234", "database":"broodbox_results", "port":3306}
MAIN_DATABASE_NAME = "broodbox"
MYSQL_RESULRS_TABLE_PREFIX = "ml_results"
HIVE_DETAILS_TABLE_PREFIX ="hive_details"
ANALYTICS_DATA_TABLE_PREFIX = "analytics"



# ## Database functions

# In[54]:

def create_mysql_table(dataset, table_name, credentials=MYSQL_CREDENTIALS_RESULTS):
    
    """ This function creates a table in mysql database using pandas dataframe"""
    
    engine = create_engine(f'mysql+mysqlconnector://{credentials["user"]}:{credentials["password"]}@{credentials["host"]}:{credentials["port"]}/{credentials["database"]}', connect_args={"connect_timeout": 28800})
    
    if "classes" in dataset.columns:
        # Serialize lists into JSON strings
        dataset["image_name"] = dataset["image_name"].apply(json.dumps)
        dataset["image_name"] = dataset["image_name"].str.replace('"', '')
        dataset["classes"] = dataset["classes"].apply(json.dumps)
        dataset["confidence"] = dataset["confidence"].apply(json.dumps)

    dataset.to_sql(table_name, con=engine, if_exists='replace', index=False)
    engine.dispose()

    
def delete_data(table_name,area_code,location_code,credentials=MYSQL_CREDENTIALS_RESULTS):  
    """This function will delete the raws where the area_code and location_code matches."""
    
    # Connect to the MySQL server
    connection = mysql.connector.connect(
        host=credentials["host"],
        user=credentials["user"],
        password=credentials["password"],
        database=credentials["database"]
    )
    
    cursor = connection.cursor()
    
    insert_sql = f"""
    DELETE FROM {table_name} WHERE area_code='{area_code}' AND location_code='{location_code}'
    """
    cursor.execute(insert_sql)
    connection.commit()
    cursor.close()
    connection.close()


def insert_multiple_raws(table_name, data, credentials=MYSQL_CREDENTIALS_RESULTS):
    """ This function will inset multiple raws of data points to the given table.
    the insert data shoulb be a dictionary"""
    
    # Connect to the MySQL server
    connection = mysql.connector.connect(
        host=credentials["host"],
        user=credentials["user"],
        password=credentials["password"],
        database=credentials["database"]
    )
    
    cursor = connection.cursor()
    
    # Prepare the INSERT query
    insert_query = f"""
    INSERT INTO {table_name} (area_code, location_code, image_name, classes, confidence, active_frame_count)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    # dictionary as a list of tuples of data points
    insert_values =[(area_code, location_code, image_name, json.dumps(classes), json.dumps(confidence), active_frame_count) 
                    for area_code, location_code, image_name, classes, confidence, active_frame_count 
                    in list(zip(data['area_code'], data['location_code'], data['image_name'], data['classes'], data['confidence'], data['active_frame_count']))]

    # Execute the INSERT query with executemany
    cursor.executemany(insert_query, insert_values) 
    # Commit the transaction
    connection.commit()  
    cursor.close()
    connection.close()


def select_data_using_query(select_query,credentials=MYSQL_CREDENTIALS_RESULTS):
    """ This function selects data according to given query.
    Then returs list of tuples like [(area_code1, location_code1, total_active_frame_count1),.........()]"""

    # Connect to the MySQL server
    connection = mysql.connector.connect(
        host=credentials["host"],
        user=credentials["user"],
        password=credentials["password"],
        database=credentials["database"]
    )  
    cursor = connection.cursor()   
    cursor.execute(select_query)
    results = cursor.fetchall()

    connection.commit()
    cursor.close()
    connection.close()

    return results


def update_active_frame_counts(data_table_name, update_table_name, data_credentials=MYSQL_CREDENTIALS_RESULTS, update_credentials=MYSQL_CREDENTIALS_MAIN):
    """ This function first gets the results of select_active_frame_counts function and then update the hive_details_BID_FID table's total_active_frames
    column in the broodbox database"""

    select_sql = f"""
                    SELECT area_code,location_code,SUM(active_frame_count) AS total_active_frames FROM {data_table_name} GROUP BY area_code,location_code;
                """

    # gets the results of select_active_frame_counts 
    update_values = select_data_using_query(select_sql, data_credentials) 
    # Connect to the MySQL server
    connection = mysql.connector.connect(
        host=update_credentials["host"],
        user=update_credentials["user"],
        password=update_credentials["password"],
        database=update_credentials["database"]
    )  
    cursor = connection.cursor()
    # updation
    for update_data in update_values:
        active_frame_count =  update_data[2]
        area_code = update_data[0]
        location_code = update_data[1]
        # Use parameterized query to update values
        update_sql = """
                UPDATE {}
                SET total_active_frames = %s
                WHERE area_code = %s AND location_code = %s
            """.format(update_table_name)

        cursor.execute(update_sql, (active_frame_count, area_code, location_code))

    connection.commit()
    cursor.close()
    connection.close()


def string_lists_to_numeric(string_list):
    """" When this fucntion is given string like this : '[10, 10, 4, 4];[10, 8];[10, 4, 4, 10];[10, 4, 10, 4];[0, 7, 6, 6];[8, 8, 6, 4];[7, 6, 6];[9]' 
    it weill convert like this : [10, 10, 4, 4, 10, 8, 10, 4, 4, 10, 10, 4, 10, 4, 0, 7, 6, 6, 8, 8, 6, 4, 7, 6, 6, 9] list"""
    split_string = string_list.split(";")
    list_srtings = [ast.literal_eval(str_element) for str_element in split_string ]
    preprocesses_results_list = []
    for sublist in list_srtings:
        preprocesses_results_list.extend(sublist)
    
    return preprocesses_results_list

def return_new_active_frme_counts(results_table_name,hive_table_name,main_database):
    """ This function select area code, location code, total beehives, classes and total active frames from ml results and hive detials tables and then calculate 
        new total active frames based on number of total beehives and ml predcted number of beehives. finally returs the results as list of lists like
        [[area code 1, location code 1, new total active frames 1], [area code 2, location code 2, new total active frames 2],................]  """

    select_sql = f"""
        SELECT {results_table_name}.area_code, {results_table_name}.location_code, GROUP_CONCAT({results_table_name}.classes SEPARATOR ';'), 
        {main_database}.{hive_table_name}.total_beehives, {main_database}.{hive_table_name}.total_active_frames
        FROM {results_table_name} INNER JOIN {main_database}.{hive_table_name} 
        ON {results_table_name}.area_code = {main_database}.{hive_table_name}.area_code AND {results_table_name}.location_code = {main_database}.{hive_table_name}.location_code
        GROUP BY {results_table_name}.area_code, {results_table_name}.location_code;
    """

    # get and preprocess the results to get [(area code 1, location code 1, total beehives 1, totoal active frames 1, class list 1),.........] list
    raw_results = select_data_using_query(select_sql)
    preprocessed_results = [(data_point[0],data_point[1],data_point[3],data_point[4],string_lists_to_numeric(data_point[2])) for data_point in raw_results]

    # go throught each result
    final_results = []
    for data_point in preprocessed_results:
        single_results = [[0] for _ in range(3)]
        # extract data form results
        area_code  = data_point[0]
        location_code = data_point[1]
        currnet_total_active_frames = data_point[3]
        class_list =  data_point[4]
        number_of_given_beehives = len(class_list)
        number_of_actual_beehives = data_point[2]
        if number_of_given_beehives == 0:
            mean_clas = 0
        else:
            mean_clas = round(statistics.mean(class_list))

        # update single point data
        single_results[0] = area_code
        single_results[1] = location_code
        # update single point active frame cound according to mean of classes of a givne location
        hive_diffenece = number_of_actual_beehives - number_of_given_beehives
        if hive_diffenece == 0:
            pass
        elif hive_diffenece > 0 or hive_diffenece < 0:
            single_results[2] =  currnet_total_active_frames + mean_clas*hive_diffenece

        final_results.append(single_results)
    
    return final_results

def update_active_frame_counts_single_table(update_table_name, update_values, update_credentials=MYSQL_CREDENTIALS_MAIN):
    """ This function updates total active frames column in the given table using given list of values like :
     [[area code 1, location code 1, new total active frames 1], [area code 2, location code 2, new total active frames 2],................] """

    # Connect to the MySQL server
    connection = mysql.connector.connect(
        host=update_credentials["host"],
        user=update_credentials["user"],
        password=update_credentials["password"],
        database=update_credentials["database"]
    )  
    cursor = connection.cursor()
    # updation
    for update_data in update_values:
        area_code = update_data[0]
        location_code = update_data[1]
        active_frame_count =  update_data[2] 
        # Use parameterized query to update values
        update_sql = """
                UPDATE {}
                SET total_active_frames = %s
                WHERE area_code = %s AND location_code = %s
            """.format(update_table_name)

        cursor.execute(update_sql, (active_frame_count, area_code, location_code))

    connection.commit()
    cursor.close()
    connection.close()


# ## Analytics functions 

def get_global_ranking(number_list):
    """ This function returns list of ranks of maximun to minimun according to given numeric list.
    ex: input = [15,11,16,17,18] then the output is = [4,5,3,2,1]. this is used to get the global ranking list of each hive location according to location richness
    of each hive location."""
    # sorte the given list in decending order
    sorted_number_list = sorted(set(number_list), reverse=True)
    rank_list = []

    for num in number_list:
        rank = sorted_number_list.index(num)+1
        rank_list.append(rank)

    return rank_list

def get_local_ranking(tuple_list,location_richness):
    """ This function get list of tuples that contained area code,location code,total_beehives,total_active_frames as tuple_list parameter.
    And location richness list as another parameter. then this will returns local ranking list according to the location richness of each hive location."""
    # This dict contains each area code as key and its richness values as a list 
    area_richness_dic = dict()
    for i in range(len(tuple_list)):
        area = tuple_list[i][0]
        richness = location_richness[i]

        if area not in area_richness_dic:
            area_richness_dic[area] = []  
        area_richness_dic[area].append(richness)
    # This dict contains each area code as key and its local ranking of richness according to the area code
    local_ranking_dict = dict()    
    for key,value in area_richness_dic.items():
        if key not in local_ranking_dict:
            local_ranking_dict[key] = get_global_ranking(value)
    # this list contains local_ranking_dict dictionary's values as one extended list 
    local_ranking_list = []
    for key,value in local_ranking_dict.items():
        local_ranking_list.extend(value)
    
    return local_ranking_list

def create_analytics_table(BID,FID,credentials=MYSQL_CREDENTIALS_RESULTS):
    """ This function cretes analytics table by using hive details table and ml results tables."""

    data_table_name = f"hive_details_{BID}_{FID}"
    select_sql = f"""
                        SELECT area_code,location_code,total_beehives,total_active_frames FROM {data_table_name};
                    """
    # get hive detail as list of tuples
    results = select_data_using_query(select_sql, credentials=MYSQL_CREDENTIALS_MAIN)
    # extract column data for the analytics table
    location_richness =  [round((res_tuple[3]/(res_tuple[2]*MEAN_FRAMES_PER_HIVE))*100,2) for res_tuple in results]
    global_ranking = get_global_ranking(location_richness)
    local_ranking = get_local_ranking(results,location_richness)
    area_codes = [result[0] for result in results]
    location_codes = [result[1] for result in results]
    total_beehives = [result[2] for result in results]
    total_active_frames = [result[3] for result in results]
    # creates dataframe 
    data_dic = {"area_code":area_codes, "location_code":location_codes, "total_beehives":total_beehives, "total_active_frames":total_active_frames, "location_richness":location_richness, "global_ranking":global_ranking,"local_ranking":local_ranking}
    dataframe  = pd.DataFrame(data_dic)
    # creates analytics table
    analytics_table_name = f"{ANALYTICS_DATA_TABLE_PREFIX}_{BID}_{FID}"
    create_mysql_table(dataframe, analytics_table_name,credentials)

def update_total_active_frames_according_to_numberof_actual_hives(ml_results_table_name, hive_details_table_name, analytics_table_name,
                                                                   mysql_credentials_results, mysql_credentials_main, main_database_name = MAIN_DATABASE_NAME,):
    """ This function call the functions wich are responsible to update the active frame counts based on total beehives and ml predicted beehives.
    this function updats the analytics and hive detials talbe's total active frame column which is used to calculate the pollination maps."""

    new_active_frme_counts = return_new_active_frme_counts(ml_results_table_name, hive_details_table_name, main_database_name)
    update_active_frame_counts_single_table(analytics_table_name, new_active_frme_counts, mysql_credentials_results)
    update_active_frame_counts_single_table(hive_details_table_name, new_active_frme_counts, mysql_credentials_main)


# ## ML prediction functions 

# In[55]:


def collect_area_location_codes(BID,FID):
    
    """ This function returns a dict that contained each area code as keys and location codes of each area code as values.
     the output format is {"area_code1":[list of location codes of that area1],.....,} """
    
    url = f"{BATABASE_API_BASE_URL}/hive/area-location-codes/{BID}/{FID}"
    response = requests.get(url)
    # this contaied all area and loation codes separately as lists of a given farm
    data = response.json()
    area_codes = data["area_codes"]

    url_all = f"{BATABASE_API_BASE_URL}/hive/{BID}/{FID}"
    response_all = requests.get(url_all)
    # this contained all the hive details of given farm
    data_all = response_all.json()

    # this dict is the requried output format. it should contained as {"area_code":[list of location codes of that area]}
    codes_dict = dict()
    for area_code in area_codes:
        codes_dict[area_code] = []

        for location in data_all["hive_details"]:

            if location["area_code"]==area_code:
                codes_dict[area_code].append(location["location_code"])
                
    return codes_dict 


def orientaion_correction(img):
    
    """This function checks the given image rotated (480*640 ---> 640*480) or not. 
    if rotated then it rotate again to oraginal format (480*640 or 640*480 ) and returns the image.
    if not rotated then it returns the original image."""
    
    # getexif attribute is a method used to retrieve Exif (Exchangeable image file format) metadata from the image.
    if hasattr(img, '_getexif') and img._getexif():
        exif_data = img._getexif()
        # 274 represents the Exif tag for orientation
        orientation = exif_data.get(274) 

        # If orientation is 6, rotate clockwise by 90 degrees
        if orientation == 6:  
            img = img.transpose(method=Image.ROTATE_270)

    return img


def create_results_folder_tree(BID,FID,codes_dict):
    
    """ This function will creates object detection results saving folder structure like data folder"""
    
    # get connection with s3 and create resluts folder
    s3_client = boto3.client('s3', aws_access_key_id= ACCESS_KEY_ID,
                    aws_secret_access_key=SECRET_ACCESS_KEY_ID)
    s3_client.put_object(Bucket=BUCKET_NAME, Key=f"images_{BID}/results_{FID}/")

    for area_code,location_list in codes_dict.items():
        s3_client.put_object(Bucket=BUCKET_NAME, Key=f"images_{BID}/results_{FID}/{area_code}/")

        for location_code in location_list:
            s3_client.put_object(Bucket=BUCKET_NAME, Key=f"images_{BID}/results_{FID}/{area_code}/{location_code}/")
            
def is_folder_exist(folder,bucket):
    
    """ If given folder exist bucket then this will returns true. other wise false."""
    
    bucket_list = list(bucket.objects.all().filter(Prefix=f"{folder}/"))
    if len(bucket_list) >=1:
        return True
    else:
        return False
    
    
def upload_image(RESULT_IMG_PATH,save_path,bucket_name):
    
    """ This function save image at given save_path at s3 bucket"""
    
    s3_client = boto3.client('s3', aws_access_key_id= ACCESS_KEY_ID,
                    aws_secret_access_key=SECRET_ACCESS_KEY_ID)
    
    with open(RESULT_IMG_PATH, "rb") as f:
        s3_client.upload_fileobj(f, bucket_name, save_path)


# In[56]:


def get_predctions_forall_locations(BID,FID,model=MODEL):
    
    """ This function loads all the images form s3 bucket and call the ML model and get predcitions.
    then saved those predictons in dict and creates folder treee at s3 bucket and then saved the outcome images at s3.
    then create a mysql table and save the data into it.
    finally returns the results dict.
    this function should use first time of all the farm image uploaded, or user wants to get predcitons of all the images
    again form the begining"""
    
    # get the connection with s3 bucket
    s3 = boto3.resource('s3',
                    aws_access_key_id=ACCESS_KEY_ID,
                    aws_secret_access_key=SECRET_ACCESS_KEY_ID)

    bucket = s3.Bucket(BUCKET_NAME)
    
     
    # get area-location code dict
    codes_dict = collect_area_location_codes(BID,FID)
    # main result
    results_dic = {"area_code":[], "location_code":[],"image_name":[], "classes":[], "confidence":[], "active_frame_count":[]}
    # checks and create results folder tree to save the resulting images
    folder = f"images_{BID}/results_{FID}"
    tree_exist = is_folder_exist(folder,bucket)
    if not tree_exist:
        create_results_folder_tree(BID,FID,codes_dict)
        
    #print(f"images of BID:{BID}, FID:{FID}")
    #count = 0

    for area_code,location_list in codes_dict.items():
        for location_code in location_list:
            
            # load image paths
            objects = list(bucket.objects.all().filter(Prefix=f"images_{BID}/data_{FID}/{area_code}/{location_code}/"))
            objects = objects[1:]
            
            # if images are there
            if len(objects) >=1:
                #print(f"AREA:{area_code} and LOCATION:{location_code}")
                for folder in objects:
                    results_dic["area_code"].append(area_code)
                    results_dic["location_code"].append(location_code) 

                    # read the image data from S3 bucket directly into memory
                    img_data = bucket.Object(folder.key).get().get('Body').read()
                    # convert image data into PIL image object
                    img = Image.open(BytesIO(img_data))
                    # rotate correction
                    img =  orientaion_correction(img) 
                    img.save(TOBE_PREDICT_IMAGE_PATH)
                    
                    # get predictions
                    prediction = model.predict(TOBE_PREDICT_IMAGE_PATH, save=True, conf=CONFIDENCE_LEVEL, exist_ok=True)
                    for results in prediction:
                        boxes = results.boxes
                    classes = [int(item.item()) for item in boxes.cls]
                    confidences = [round(item.item(),2) for item in boxes.conf]
                    
                    results_dic["classes"].append(classes)
                    results_dic["confidence"].append(confidences)
                    results_dic["active_frame_count"].append(sum(classes))
                    image_name = folder.key.split('/')[-1]
                    results_dic["image_name"].append(image_name)
                    # upload the output image to s3 
                    save_path = f"images_{BID}/results_{FID}/{area_code}/{location_code}/{image_name}"
                    upload_image(RESULT_IMG_PATH,save_path,BUCKET_NAME)
     
            else:
                results_dic["area_code"].append(area_code)
                results_dic["location_code"].append(location_code) 
                results_dic["image_name"].append([])
                results_dic["classes"].append([])
                results_dic["confidence"].append([]) 
                ## USED RANDOM VALUE TO CLACULATE POLLINATION MAP. WHEN ACTUAL CASE FILL THIS, USING np.NaN 
                results_dic["active_frame_count"].append(random.randint(20, 100))

    # creates a mysql table and store the results
    dataset = pd.DataFrame(results_dic)
    table_name = f"{MYSQL_RESULRS_TABLE_PREFIX}_{BID}_{FID}"
    create_mysql_table(dataset, table_name, credentials=MYSQL_CREDENTIALS_RESULTS)
    # update hive details table to pollination map
    hive_details_table_name = f"{HIVE_DETAILS_TABLE_PREFIX}_{BID}_{FID}"
    update_active_frame_counts(table_name, hive_details_table_name)
    # create analytics table
    create_analytics_table(BID,FID)
    # update the total active frames according to total beehives and ml predced beehives in ml result and analytics tables
    analytics_table_name = f"{ANALYTICS_DATA_TABLE_PREFIX}_{BID}_{FID}"
    update_total_active_frames_according_to_numberof_actual_hives(table_name, hive_details_table_name, analytics_table_name, MYSQL_CREDENTIALS_RESULTS, MYSQL_CREDENTIALS_MAIN)
        
    return results_dic


# In[57]:


def get_predtictions_specific_location(BID,FID,area_code,location_code,model=MODEL):
    """This function get load images, get predictions, save resulting images and returns 
    resulting dictionary of data for a given area_code and location_code only. (only for a single location).
    This function should use when a user update the images form a specific hive location."""
    # get the connection with s3 bucket
    s3 = boto3.resource('s3',
                        aws_access_key_id=ACCESS_KEY_ID,
                        aws_secret_access_key=SECRET_ACCESS_KEY_ID)

    bucket = s3.Bucket(BUCKET_NAME)
    # main result
    results_dic = {"area_code":[], "location_code":[],"image_name":[], "classes":[], "confidence":[], "active_frame_count":[]}
    # load image paths
    objects = list(bucket.objects.all().filter(Prefix=f"images_{BID}/data_{FID}/{area_code}/{location_code}/"))
    objects = objects[1:]
    
    # if images are there
    if len(objects) >=1:

        #print(f"AREA:{area_code} and LOCATION:{location_code}")
        for folder in objects:
            results_dic["area_code"].append(area_code)
            results_dic["location_code"].append(location_code) 

            # read the image data from S3 bucket directly into memory
            img_data = bucket.Object(folder.key).get().get('Body').read()
            # convert image data into PIL image object
            img = Image.open(BytesIO(img_data))
            # rotate correction
            img =  orientaion_correction(img) 
            img.save(TOBE_PREDICT_IMAGE_PATH)

            # get predictions
            prediction = model.predict(TOBE_PREDICT_IMAGE_PATH, save=True, conf=CONFIDENCE_LEVEL, exist_ok=True)
            for results in prediction:
                boxes = results.boxes
            classes = [int(item.item()) for item in boxes.cls]
            confidences = [round(item.item(),2) for item in boxes.conf]

            results_dic["classes"].append(classes)
            results_dic["confidence"].append(confidences)
            results_dic["active_frame_count"].append(sum(classes))
            image_name = folder.key.split('/')[-1]
            results_dic["image_name"].append(image_name)

            save_path = f"images_{BID}/results_{FID}/{area_code}/{location_code}/{image_name}"
            upload_image(RESULT_IMG_PATH,save_path,BUCKET_NAME)
    else:
        results_dic["area_code"].append(area_code)
        results_dic["location_code"].append(location_code) 
        results_dic["image_name"].append([])
        results_dic["confidence"].append([]) 
        results_dic["classes"].append([])

        ## USED RANDOM VALUE TO CLACULATE POLLINATION MAP. WHEN ACTUAL CASE FILL THIS, USING np.NaN 
        results_dic["active_frame_count"].append(random.randint(20, 100)) 
        
    # update the ml results table by deleteing old data and inserting new data for the specified location
    table_name = f"{MYSQL_RESULRS_TABLE_PREFIX}_{BID}_{FID}"    
    delete_data(table_name,area_code,location_code,credentials=MYSQL_CREDENTIALS_RESULTS)
    insert_multiple_raws(table_name, results_dic, credentials=MYSQL_CREDENTIALS_RESULTS)
    # update hive details table to pollination map
    hive_details_table_name = f"{HIVE_DETAILS_TABLE_PREFIX}_{BID}_{FID}"
    update_active_frame_counts(table_name, hive_details_table_name)
    # update analytics table
    create_analytics_table(BID,FID)
    # update the total active frames according to total beehives and ml predced beehives in ml result and analytics tables
    analytics_table_name = f"{ANALYTICS_DATA_TABLE_PREFIX}_{BID}_{FID}"
    update_total_active_frames_according_to_numberof_actual_hives(table_name, hive_details_table_name, analytics_table_name, MYSQL_CREDENTIALS_RESULTS, MYSQL_CREDENTIALS_MAIN)

    return results_dic


# In[58]:


#result = load_images_get_predctions(BID,FID)


# In[59]:


#dataset = pd.DataFrame(result)


# In[60]:


#dataset.head(50)


# In[61]:


"""s3 = boto3.resource('s3',
                    aws_access_key_id= 'AKIA4EQ6TDBWJ7BM5DK7',
                    aws_secret_access_key='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')

bucket = s3.Bucket('beehive-thermal-images-testing')

# specify the image and its key in the bucket
image_key = f"data-{BID}-{FID}/1/11139/FLIR0222.jpg"

# read the image data from S3 bucket directly into memory
img_data = bucket.Object(image_key).get().get('Body').read()

# convert image data into PIL image object
img = Image.open(BytesIO(img_data))

# do something with the image object, e.g. display it
#img.show()
img.save("new_img.png")

img"""

