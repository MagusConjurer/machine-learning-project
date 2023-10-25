# machine-learning-project
This is a machine learning project solution developed by Cameron Davis for CS5350/6350 at the University of Utah.  

The project details can be found here: [Kaggle - Predicting Income Level](https://www.kaggle.com/competitions/income2023f/overview)

### Midterm Progress

I started off by looking into the scikit-learn library. As a starting place, I decided to use AdaBoost which we had recently covered in class.  

The primary time was spent working through getting the data loaded and formatted in the way that the sklearn methods expected. Once that was working I was able to put together a function to write the resulting predictions from the test data to a csv file for submission. My first working submission received an AUROC score of 0.69186 (calculated with approximately 50% of the test data) put me in the top 5 on the leaderboard. This is satisfactory for me until we cover more advanced algorithms that I can implement for the final submission. 

#### How to run
Download the project and then run the shell file in the top directory:  

    ./run.sh

Results will be printed in the console similar to this:  

    Processing training data
    Time elapsed : 0:00:19.954528
    Processing test data
    Time elapsed : 0:00:19.052239
    Fitting the data using default AdaBoost
    Writing predictions to file: predictions/prediction-25Oct-1030.csv

### Final Progress
