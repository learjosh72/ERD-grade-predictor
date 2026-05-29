# ERD-grade-predictor
# How To Run
1. make sure all included files are downloaded and all accessible within the same directory
2. make sure all packages and libraries are downloaded in the environment
3. edit "mainfilestage2.py" to represent the true paths of the datasets and grades
4. put the links to the training dataset in the variables ds1path (& ds2path if two datasets)
5. put the links to the training dataset in the variables ds1tpath (& ds2tpath if two datasets)
6. edit the "gradespath" variable to be the path to the grades of the training set
7. if running on less than 2 datasets, remove the necessary approach calls
8. make sure the "testing" variable is set to False if trying to get new grades
9. run the file named "mainfilestage2.py"
10. you can do this from the command line with "python3 mainfilestage2.py" or by simply running the file in an ide
11. this will produce the necessary output files containing the grades per approach in the directory
