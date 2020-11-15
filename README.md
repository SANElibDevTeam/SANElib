# SANElib Prototype

The goal of this work is to make this existing SQL-based density estimation technique, whose prototypical implementation in SQL is already available, usable with a wrapper for the R or Python environment, and to evaluate and compare its speed and predictive performance. The expected result of the project is a R or Python package, which allows the following functions:
* **Input**: A data table, which is larger than the working memory, can be input as parameter. This table is processed based on a file or a database. 
*	**Training phase**: the input data table is quantized (equal size, N_TILE vs. equal width, WIDTH_BUCKET) and indexed in a SQL-DBMS using SQL. This quantized index then represents an in-database model for density estimation.
*	**Prediction phase**: based on this model For a new record or set of records, it is possible to determine a multivariate density estimation for a new data record or a whole set of records for selected columns of the table.
*	**Visualization**: the density function can be visualized with R or Python relatively easily in 1D and 2D.
*	**Embedded DBMS**: By default, an embedded DBMS is used for the calculation (e.g. H2 or SQLite) and the calculation is done locally on the client.
*	**Code-to-Data**: Any connection to a database server can be configured via a parameter, so that a server-side calculation in SQL can be performed.
*	**Distribution**: The R or Python package is well documented, works without errors and is ready for Publication on CRAN or PyPi.

