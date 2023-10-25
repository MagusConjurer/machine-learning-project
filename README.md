# machine-learning-project
This is a machine learning project solution developed by Cameron Davis for CS5350/6350 at the University of Utah.  

The project details can be found here: [Kaggle - Predicting Income Level](https://www.kaggle.com/competitions/income2023f/overview)

### Midterm Progress

I started off by looking into the scikit-learn library. As a starting place, I decided to use AdaBoost which we had recently covered in class.  

The primary time was spent working through getting the data loaded and formatted in the way that the sklearn methods expected. Once that was working I was able to put together a function to write the resulting predictions from the test data to a csv file for submission. My first working submission received an AUROC score of 0.69186 (calculated with approximately 50% of the test data) put me in the top 5 on the leaderboard. 

For my second submission, I moved on to the Random Forest classifier that we had also covered when talking about ensemble learning. Since everything else was already in place, it only required swapping out a few references to AdaBoost before I had a new list of predictions. This submission received an AUROC score of 0.70328. To me this confirms the improvement that we have seen as we cover more and more classification methods.

For the final progress I plan to use some of the algorithms that we are discussing currently and over the next month.

#### How to run
Download the project and then run the shell file in the top directory:  

    ./run.sh

Results will be printed in the console similar to this:  

    Running Kaggle - Predict Income Level code

    Processing training data
    Time elapsed : 0:00:06.791361
    Processing test data
    Time elapsed : 0:00:06.519691
    Fitting the data using default Random Forests
    Writing predictions to file: predictions/prediction-25Oct-1054.csv


### Final Progress
