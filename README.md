# Random-Pair Nested Dichotomies

This repository contains Java source files for random-pair nested dichotomies (RPND), and nested dichotomies based on clustering (NDBC) [1]. Class-balanced and standard nested dichotomies can be found in the `ensemblesOfNestedDichotomies` package in WEKA.

To compile:

    javac -cp "path_to_weka.jar:." weka/classifiers/meta/nestedDichotomies/*.java
  
To use in WEKA:

    java -cp "path_to_weka.jar:." weka.gui.Main

[1]: Duarte-Villaseñor, Miriam Mónica, et al. "Nested Dichotomies Based on Clustering." Progress in Pattern Recognition, Image Analysis, Computer Vision, and Applications. Springer Berlin Heidelberg, 2012. 162-169.
