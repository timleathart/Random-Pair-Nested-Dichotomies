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

[1]: Leathart, Tim, Bernhard Pfahringer, and Eibe Frank. "Building Ensembles of Adaptive Nested Dichotomies with Random-Pair Selection." arXiv preprint arXiv:1604.01854 (2016).

[2]: Duarte-Villaseñor, Miriam Mónica, et al. "Nested dichotomies based on clustering." Iberoamerican Congress on Pattern Recognition. Springer Berlin Heidelberg, 2012.


