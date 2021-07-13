# SANElib - Contribute

If you're interested to contribute to this project note the following information:

- Project Structure:

  |- sanelib.py (SANElib factory)

  |- config.py (Database configuration parameters)

  |- requirements.txt (Python requirements)

  |- lib (Main library)

  -----|- linear_regression

  -----|- mdh

  -----|- ...

  |- util (General utilities)

  -----|- database (Database connection utilities)

  |- example_datasets (Example datasets for implementations)

  |- images (Documentation images)			

- util/database.py DBConnection

  - **execute**(query)

    Executes query inside the database.

    `Parameters`

    *query: SQL query (String/Text)*.

  - **execute_query**(query)

    Executes query inside the database and returns it's results.

    `Parameters`

    *query: SQL query (String/Text)*.

    `Return`: *Returns the result of the query provided*. 	 									

- Quick Start (please also refer to existing implementations):

1. Create Issue

2. ![Create Issue](C:\Users\patri\Documents\Patrick\HSLU\BAA\Code\linear_regression\SANElib\images\Issues.png)

3. Create new branch from dev named like issue.

4. You can create a new branch directly on Git, or you can create a new branch the following:

5. ```bash
   git checkout dev
   git pull
   git branch issue
   git checkout issue
   ```

6. Commit your changes and push to issue branch

7. Checkout to dev branch

8. ```bash
   git checkout dev
   ```

9. Pull newest dev branch

10. ```bash
    git pull
    ```

11. Checkout to issue branch

12. ```bash
    git checkout issue
    ```

13. Add new changes from dev branch to feature branch

14. ```bash
    git merge dev
    ```

15. ![git merge main](C:\Users\patri\Documents\Patrick\HSLU\BAA\Code\linear_regression\SANElib\images\GitMerge.png)

16. Commit and push all changes to origin

17. ```bash
    git commit -m "Merged changes on dev to issue branch"
    git push
    ```

18. Create pull request to merge issue branch to dev branch.

    ![Create Pull request](C:\Users\patri\Documents\Patrick\HSLU\BAA\Code\linear_regression\SANElib\images\PullRequest.PNG)

19. Choose the branch you want to merge on the right (here kmeans) and the dev branch on the left.

20. Create a new folder under lib with the name of your implementation.

21. Create a new folder "sql_templates" in your implementation folder. Make a separate .py file for all database types the algorithm supports.

22. Create your main python file (e.g. "implementation_x").

23. Add your implemenation_x.py to init.py inside the lib folder.

24. Extend sanelib.py with your implementation (x = lib.x.x(db))

- #### Database functions (util)

  - **get_column**(database, table)

    `Parameters`

    *database: Database in which the table to be queried lies.* 

    *table: Table to query the column names from.*

    `Return`: *Returns all column names of a table in a numpy array.*

  - **get_number_of_rows**(database, table)

    `Parameters`

    *database: Database in which the table to be queried lies.*

    *table: Table to query the number of rows from.*

    `Return`: *Returns the number of rows listed in a table.*

  - **multiply_matrices**(database, table_a, table_b, result_table_name)

    Calculates the result of the matrix multiplication from AB (table_a*table_b) and stores the result in matrix form into result_table_name. 

    !Important: both table_a and table_b must contain a column named id and this id must represent the place of each row in the matrix.

    `Parameters`

    *database: Database the tables table_a and table_b are available.*

    *table_a: Left input table.*

    *table_b: Right input table.*

    *result_table_name: Name of the table to be created/replaced, used to store the result matrix.*