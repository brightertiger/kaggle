### Package Dependencies
* numpy
* pandas
* lightgbm
* catboost
* scikit-learn

### Directory Structure

* **code**: has all the notebooks. <u>They should be run in the order in which they appear<u>.
	* 00-*.ipynb: data cleaning + driver file
	* 01-*.ipynb: Feature Engineering
	* 02-*.ipynb: Merging the features to create training, validation and full dataset
	* 03-*.ipynb: Model-1 Training and Scoring  
	* 04-*.ipynb: Model-2 Training and Scoring  
	* 05-*.ipynb: Model-3 Training and Scoring
	* 06-*.ipynb: Blending
	* **score.csv** with be created in this directory and is the final solution file

* **data**: should have all the data files. is currently empty. 
	* **data**: put the original CSV files here before running the code . 
	* feature: has the various CSV files created during execution of 01*.ipynb 
	* model: has the merged dataset from 02-*.ipynb and model files
	* score: has the scored CSV files from the three models for belnding


