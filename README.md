Random-Pair Nested Dichotomies
==============================

This repository contains Java source files for random-pair nested dichotomies (RPND) [1], and nested dichotomies based on clustering (NDBC) [2]. Class-balanced and standard nested dichotomies can be found in the `ensemblesOfNestedDichotomies` package in WEKA.

Java 
----

To compile:

    javac -cp "path_to_weka.jar:." weka/classifiers/meta/nestedDichotomies/*.java
  
To use in WEKA:

    java -cp "path_to_weka.jar:." weka.gui.Main
    
Python
------

Work in progress. Currently, you can make nested dichotomies with completely random selection, and class balanced.

References
----------

[1]: Leathart, T., Pfahringer, B., & Frank, E. (2016). Building Ensembles of Adaptive Nested Dichotomies with Random-Pair Selection. arXiv preprint arXiv:1604.01854.

[2]: Duarte-Villaseñor, M. M., Carrasco-Ochoa, J. A., Martínez-Trinidad, J. F., & Flores-Garrido, M. (2012, September). Nested dichotomies based on clustering. In Iberoamerican Congress on Pattern Recognition (pp. 162-169). Springer Berlin Heidelberg.


