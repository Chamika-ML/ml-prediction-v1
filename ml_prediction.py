#!/usr/bin/env python
# coding: utf-8

# In[27]:


import boto3
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import requests
import json
import pandas as pd
import numpy as np
import random
 


import ultralytics
ultralytics.checks()
from ultralytics import YOLO

#!pip install mysql-connector-python
import mysql.connector
#!pip install sqlalchemy
from sqlalchemy import create_engine, inspect

BID = "B456"
FID = "123"
TOBE_PREDICT_IMAGE_PATH = f"./images/need_to_predict_{BID}_{FID}.png"
RESULT_IMG_PATH =  f"./runs/detect/predict/need_to_predict_{BID}_{FID}.png"
MODEL = YOLO("./model//best.pt")
CONFIDENCE_LEVEL = 0.5
ACCESS_KEY_ID = "AKIA4EQ6TDBWJ7BM5DK7"
SECRET_ACCESS_KEY_ID = "9zO14I1rRtGmiSBKEc2X70Inc101SpDL7BsWrtqD"

MYSQL_CREDENTIALS = {"host":"127.0.0.1", "user":"dilshan", "password":"1234", "database":"broodbox_results", "port":3306}
MYSQL_RESULRS_TABLE_PREFIX = "ml_results"


# ## Database functions

# In[ ]:


def create_mysql_table(dataset, table_name, credentials=MYSQL_CREDENTIALS):
    
    "this function creates a table in mysql database using pandas dataframe"
    
    engine = create_engine(f'mysql+mysqlconnector://{credentials["user"]}:{credentials["password"]}@{credentials["host"]}:{credentials["port"]}/{credentials["database"]}', connect_args={"connect_timeout": 28800})

    dataset.to_sql(table_name, con=engine, if_exists='replace', index=False)
    
    engine.dispose()


# ## ML prediction functions 

# In[10]:


def collect_area_location_codes(BID,FID):
    
    """ This function returns a dict that contained each area code as keys and location codes of each area code as values.
     the output format is {"area_code1":[list of location codes of that area1],.....,} """
    
    url = f"http://ec2-54-206-119-102.ap-southeast-2.compute.amazonaws.com:5000/hive/area-location-codes/{BID}/{FID}"
    response = requests.get(url)
    # this contaied all area and loation codes separately as lists of a given farm
    data = response.json()
    area_codes = data["area_codes"]

    url_all = f"http://ec2-54-206-119-102.ap-southeast-2.compute.amazonaws.com:5000/hive/{BID}/{FID}"
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
    s3_client.put_object(Bucket="beehive-thermal-images-testing", Key=f"results_{BID}_{FID}/")

    for area_code,location_list in codes_dict.items():
        s3_client.put_object(Bucket="beehive-thermal-images-testing", Key=f"results_{BID}_{FID}/{area_code}/")

        for location_code in location_list:
            s3_client.put_object(Bucket="beehive-thermal-images-testing", Key=f"results_{BID}_{FID}/{area_code}/{location_code}/")
            
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


# In[28]:


def load_images_get_predctions(BID,FID,model=MODEL):
    
    """ This function loads images form s3 bucket and call the ML model and get predcitions.
    then saved those predictons in dict and creates folder treee at s3 bucket and then saved the outcome images at s3.
    then create a mysql table and save the data into it.
    finally returns the results dict"""
    
    bucket_name = "beehive-thermal-images-testing"
    # get the connection with s3 bucket
    s3 = boto3.resource('s3',
                    aws_access_key_id=ACCESS_KEY_ID,
                    aws_secret_access_key=SECRET_ACCESS_KEY_ID)

    bucket = s3.Bucket(bucket_name)
    
     
    # get area-location code dict
    codes_dict = collect_area_location_codes(BID,FID)
    # main result
    results_dic = {"area_code":[], "location_code":[], "classes":[], "confidence":[], "total_active_frames":[]}
    # checks and create results folder tree to save the resulting images
    folder = f"results_{BID}_{FID}"
    tree_exist = is_folder_exist(folder,bucket)
    if not tree_exist:
        create_results_folder_tree(BID,FID,codes_dict)
        
    #print(f"images of BID:{BID}, FID:{FID}")
    #count = 0

    for area_code,location_list in codes_dict.items():
        for location_code in location_list:
            
            # load image paths
            objects = list(bucket.objects.all().filter(Prefix=f"data-{BID}-{FID}/{area_code}/{location_code}/"))
            objects = objects[1:]
            
            # if images are there
            if len(objects) >=1:
                print(f"AREA:{area_code} and LOCATION:{location_code}")
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
                    results_dic["total_active_frames"].append(sum(classes))
                    
                    save_path = f"results_{BID}_{FID}/{area_code}/{location_code}/{folder.key.split('/')[-1]}"
                    upload_image(RESULT_IMG_PATH,save_path,bucket_name)
     
            else:
                results_dic["area_code"].append(area_code)
                results_dic["location_code"].append(location_code) 
                results_dic["total_active_frames"].append([])
                results_dic["confidence"].append([])
                
                ## USED RANDOM VALUE TO CLACULATE POLLINATION MAP. WHEN ACTUAL CASE FILL THIS, USING np.NaN 
                results_dic["classes"].append(random.randint(0, 40))
            
            # creates a mysql table and store the results
            dataset = pd.DataFrame(results_dic)
            table_name = f"{MYSQL_RESULRS_TABLE_PREFIX}_{BID}_{FID}"
            create_mysql_table(dataset, table_name, credentials=MYSQL_CREDENTIALS)
    
    return results_dic


# In[12]:


#result = load_images_get_predctions(BID,FID)


# In[13]:


#dataset = pd.DataFrame(result)


# In[29]:


#dataset.head(50)


# In[15]:


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

