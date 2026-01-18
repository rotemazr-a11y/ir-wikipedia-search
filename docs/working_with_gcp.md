This manual given by course staff for the third course assigment and can bu used as guidelines for the project for how the working with gcp should look like will help you set up GCP and will cover the following: 
1.	How to redeem your credits
2.	Creating a project and check billing 
3.	Set up the environment to working with pyspark on GCP 
4.	Creating a bucket on google storage
5.	Working with Dataproc and creating a cluster
6.	Figure out the cost breakdown of your GCP usage.


Please note that we might not cover everything, but GCP documentation online is clear and easy to follow. 

** The setup will take about an hour to complete. **

Redeem credits
Before we dig in, we first need to create our GCP account. Each student has $50 to work on GCP and you should redeem your credits before getting started. 
To redeem your credits you should enter one of the following links (here). Please try the first link and if it doesn’t work enter the second one. Make sure you use your post account. 
You don’t need to insert your credit card to use the credits. However, you should make sure you have enough credits because we can’t get more.If you finish your credits before the end of the course you will be able to enter a free trial for a month and get $300 free credits, but at this point you will need to enter your credit card details. 
Redeeming your credits automatically creates your GCP account. 

Create project
At the top bar there is a “select a project” button.
 

Clicking it will open a window where you can create a new project. 
 
Name the project as you want and click create. 

Check Billing 
Using the navigation menu at the left of the top bar, access the Billing. You will arrive to this page:
 
click the “link a billing account” and connect your project to the billing account containing the credits. 
If we get to the billing overview page without getting the “link a billing account” msg, it is possible that your project is already connected to the billing account. 
In order to check this, go to “account management” and make sure your project appears on the list. The billing account should be the “Billing Account for Education”. 
 

For more information, you can access GCP documentation on billing. 

Setting up the environment
1.	Make sure to enable both the Compute Engine API and the Cloud Storage API. 
2.	Install cloud SDK following this documentation. 
3.	Create a service account (more on service accounts can be found here):
a.	Navigation menu → IAM & Admin → Service Accounts → create service account.
b.	In the “Service account name” field, enter a name. The Cloud Console fills in the “Service account ID” field based on this name. In the Service account description field, enter a description. Then click create and continue.
c.	Click the Select a role field. Under Quick access, click Basic, then click Owner. Then click continue and then Done to finish creating the service account.
d.	Create a service account key - In the Cloud Console, click the service accounts as previously mentioned, click the project you created, then click the three points manage keys → Click Keys → Click Add key, then click Create new key → Click Create. A JSON key file is downloaded to your computer → Click Close.
e.	Set the environment variable GOOGLE_APPLICATION_CREDENTIALS to the path of the JSON file that contains your service account key. This variable only applies to your current shell session, so if you open a new session, set the variable again.(Replace KEY_PATH with the path of the JSON file that contains your service account key). In the GCP assignment you will need to update this line in the code to have your own local key_path instead of the path that currently exists. 
i.	export GOOGLE_APPLICATION_CREDENTIALS="KEY_PATH"
 


Google Storage
In the course we work with wiki dumps, that are XML files. In order to save you (and us) time, we created preprocessed files that are used for the project and the 3rd assignment.
All files are in a public Google Storage Bucket, and can be accessed by you at all times. Instructions on how to access the bucket and the files are presented in assignment 3. 

Other than the public bucket we created for you that holds all the data, you will need to create your own storage bucket.
1.	Go to Cloud storage→ browser → create bucket.
2.	 Give your bucket a unique name, preferably containing your id (there can’t be two buckets with the same name on google storage). 
3.	In the location type choose Region then select us-central1. 
4.	click create

*After you created the bucket, we should upload the initialization file (.sh file) needed for assignment 3. The file can be found on moodle under assignment 3. Click on the bucket you created then click upload files and upload the relevant file through the browser that was opened.
* You also need to transfer the data to your bucket in order to use him.
1.	Click on the bucket you created and then click transfer data → transfer data in.
2.	Go to choose a source and paste the name wikidata20210801_preprocessed.
3.	Go to choose a destination and select the bucket you just created.
4.	Click on create.
After the job is finished your bucket will contain all the multistream files.


Create cluster

After we connect the billing account, we can create a cluster that we will use in the project and in the 3rd assignment. To open a cluster we will first need to enable the DataProc API. You can either find the API in the navigation menu, or you can search for it in the search bar at the top of the page. Then, click on the enable button. 
 
Now, we can open a new cluster. 
Step 1: VPC Network Configuration
1. In Google Cloud Console, click on the navigation menu (☰) in the top-left corner
2. Scroll down to "Network services" and select "VPC network"
3. Click on "VPC networks" from the left sidebar
4. Find the default network or your custom network in the list
5. Click on the network name to open its details
6. Scroll down to the "Subnet creation mode" section
7. Find the subnet in "us-central1" region
8. Click the edit (pencil) icon
9. Find "Private Google Access" and change it to "On"
10. Click "Save"

Step 2: Open Cloud Shell
1. Look at the top-right corner of the Google Cloud Console
2. Find the Cloud Shell icon (it looks like `>_`)
3. Click the icon to open Cloud Shell
4. Wait for the message "Cloud Shell provisioned" to appear
5. You should see a command prompt like `username@cloudshell:~ (project-name)$`


Step 3: Set Environment Variables
1. In Cloud Shell, Copy the command below, but replace [YOUR-PROJECT-NAME] with the name of the project you created earlier:
export REGION="us-central1"
export ZONE="us-central1-a"
export PROJECT_NAME= “:[YOUR- PROJECT-NAME]”
2. Press Enter after each command
3. Verify the variables are set by typing:
echo $REGION
echo $ZONE
echo $PROJECT_NAME
Each should display the corresponding value you set.
4. Create the Dataproc Cluster
- Copy the command below, but replace [YOUR-BUCKET-NAME] with the name of the bucket you created earlier, and Paste it into Cloud Shell:
gcloud dataproc clusters create cluster-0016 \
    --enable-component-gateway \
    --region $REGION \
    --zone $ZONE \
    --project $PROJECT_NAME \
    --master-machine-type n1-standard-4 \
    --master-boot-disk-size 100 \
    --num-workers 2 \
    --worker-machine-type n1-standard-4 \
    --worker-boot-disk-size 100 \
    --image-version 2.0-debian10 \
    --optional-components JUPYTER \
    --scopes 'https://www.googleapis.com/auth/cloud-platform' \
    --initialization-actions 'gs://[YOUR-BUCKET-NAME]/graphframes.sh'

4. Wait for the cluster creation to complete (this may take 5-10 minutes)
5. You'll see a success message when the cluster is ready


Important - The main waste of your credits is the cluster, so it is highly recommended to delete it at the end of the work, so that you don't run out of money for the project.

Run spark on GCP
Now that we have a cluster on Dataproc, we can open a jupyter notebook and start writing code. Go to Dataproc → clusters → click on the name of the cluster we just created. 
This will open a window with all the information on the cluster. Click on web interfaces and then jupyter. Now you are in the jupyter environment. We can create jupyter notebooks in the local disk folder, navigate to the home directory (‘/ Local Disk / home / dataproc’ folder). If you want to run pyspark, click new → pySpark.
If you upload the notebook instead of creating it you should change the kernel to pySpark. 
Make sure you upload both the `inverted_index_gcp.py` file and the `assignment3_gcp.ipynb` notebook to the directory (‘/ Local Disk / home / dataproc’ folder).

Perfect, now you can start writing your code. 


Figure out the cost breakdown of your usage
1.	Read this documentation on the topic and make sure you watch the video there. 
2.	In your GCP console, go to Billing → reports. Select the relevant time range, group by SKU and you'll see a detailed breakdown of costs.

Good luck! 

