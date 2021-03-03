# SANElib Prototype

The goal of this work is to implement specific ML procedures using SQL code generation, and is then made interactively available via Python using a wrapper to evaluate and compare its speed and predictive performance. The expected result of the project is a Python package, which allows the following functions:

## Multidimensional histogram (MDH) probability estimation

* **Input**: A data table, which is larger than the working memory, can be input as parameter. This table is processed based on a file or a database. 
*	**Training phase**: The input data table is quantized (equal size, N_TILE vs. equal width, WIDTH_BUCKET) and indexed in a SQL-DBMS using SQL. This quantized index then represents an in-database model for density estimation.
*	**Prediction phase**: Based on this model For a new record or set of records, it is possible to determine a multivariate density estimation for a new data record or a whole set of records for selected columns of the table.
*	**Visualization**: The density function can be visualized with R or Python relatively easily in 1D and 2D.
*	**Embedded DBMS**: By default, an embedded DBMS is used for the calculation (e.g. H2 or SQLite) and the calculation is done locally on the client.
*	**Code-to-Data**: Any connection to a database server can be configured via a parameter, so that a server-side calculation in SQL can be performed.

## More Procedures to follow

## Test Database

In order to test the SANElib library run the main.py file. We created a sql script, which creates a revised version of the Covertype dataset, where the categorical attributes have been unpivoted, and it has been formated for MySQL.
It can be downloaded here: https://doi.org/10.5281/zenodo.4562534

To be able to connect to the local database, a constants.py file must be added to the project. The constants_example.py file shows which information must be contained in the constants.py file.

