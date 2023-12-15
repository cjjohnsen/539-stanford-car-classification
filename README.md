# How to run the code
Our dataset does not fit into a GitHub repository, so it must first be downloaded. There is a link in `download_data.py` which may need to be updated in order for the data to correctly download.

## How to update the download link
In order to attain a download link, you must first go to the Kaggle dataset page: 
https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder/code

Then, inspect element and go to the Network tab: 
![Alt text](<Screenshot 2023-12-14 at 11.11.39 PM.png>)

After clicking Download on the dataset page, you will see a request whose initiator is "download": 
![Alt text](<Screenshot 2023-12-14 at 11.15.59 PM.png>)

When selecting this event, there will be a request URL that starts with "https://storage.googleapis.com/...": 
![Alt text](<Screenshot 2023-12-14 at 11.16.16 PM.png>)

Paste this link into the `SC_ZIP_URL` variable in `download_data.py` to enable downloading.

## How to train the model
To train the model, simply run `python/python3 train.py`. This will download the data (if needed) and run the training for 1000 epochs. Note that the script saves a model every 10 epochs to `/data/models`, so all 1000 epochs do not need to be run.

## How to evaluate the model
There is currently a model in `/data/models` called `model.pth` which had been trained for 110 epochs. To evaluate this model, run `python/python3 eval.py`. This will print out a list of all of the individual class accuracies, as well as generate sample images with classes and a confusion matrix. To run the evalution script on a model trained on a different number of epochs, change the name of the model that you want to run from `model_<NUM_EPOCHS>.pth` to `model.pth`.