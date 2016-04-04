/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    RandomPairND.java
 *    Copyright (C) 2016 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.meta.nestedDichotomies;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.RandomizableSingleClassifierEnhancer;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.MakeIndicator;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.classifiers.Evaluation;

import java.util.*;

/**
 * @author Tim Leathart
 *
 * Based on ClassBalancedND.java
 */
public class RandomPairND extends RandomizableSingleClassifierEnhancer {

  /** for serialization */
  static final long serialVersionUID = 5944063630650811903L;

  /** The filtered classifier in which the base classifier is wrapped. */
  protected FilteredClassifier m_FilteredClassifier;

  /** The hashtable for this node. */
  protected Hashtable<String, Classifier> m_classifiers;

  /** The percentage of the instances to train the initial classifier on */
  protected double m_subsamplePercent = 100.0;

  /** The number of random pairs to pick, where the best pair is chosen */
  protected int m_numRandomPairs = 1;

  /** The first successor */
  protected RandomPairND m_FirstSuccessor = null;

  /** The second successor */
  protected RandomPairND m_SecondSuccessor = null;

  /** The classes that are grouped together at the current node */
  protected Range m_Range = null;

  /** Is Hashtable given from END? */
  protected boolean m_hashtablegiven = false;

  /**
   * Constructor.
   */
  public RandomPairND() {

    m_Classifier = new weka.classifiers.functions.Logistic();
  }

  /**
   * String describing default classifier.
   * 
   * @return the default classifier classname
   */
  @Override
  protected String defaultClassifierString() {
    return "weka.classifiers.functions.Logistic";
  }


  /**
   * Set hashtable from END.
   * 
   * @param table the hashtable to use
   */
  public void setHashtable(Hashtable<String, Classifier> table) {

    m_hashtablegiven = true;
    m_classifiers = table;
  }

  /**
   * Generates a classifier for the current node and proceeds recursively.
   * 
   * @param data contains the (multi-class) instances
   * @param classes contains the indices of the classes that are present
   * @param rand the random number generator to use
   * @param classifier the classifier to use
   * @param table the Hashtable to use
   * @throws Exception if anything goes wrong
   */
  private void generateClassifierForNode(Instances data, Range classes,
    Random rand, Classifier classifier, Hashtable<String, Classifier> table)
    throws Exception {

    // Get the indices
    int[] indices = classes.getSelection();

    // Randomize the order of the indices
    for (int j = indices.length - 1; j > 0; j--) {
      int randPos = rand.nextInt(j + 1);
      int temp = indices[randPos];
      indices[randPos] = indices[j];
      indices[j] = temp;
    }

    int[] initialIndices = new int[2];
    System.arraycopy(indices, 0, initialIndices, 0, 2);

    Set<Integer> set1 = new HashSet<Integer>();
    Set<Integer> set2 = new HashSet<Integer>();

    set1.add(initialIndices[0]);
    set2.add(initialIndices[1]);

    if (indices.length > 2) {
      Resample resample = new Resample();
      resample.setSampleSizePercent(m_subsamplePercent);
      resample.setInputFormat(data);

      RemoveWithValues initialRwv = new RemoveWithValues();
      initialRwv.setInvertSelection(true);
      initialRwv.setInputFormat(data);
      initialRwv.setAttributeIndex("" + (data.classIndex() + 1));
      initialRwv.setNominalIndicesArr(initialIndices);

      // Filter.useFilter(data, initialRwv);

      MultiFilter multiFilter = new MultiFilter();
      multiFilter.setFilters(new Filter[] {resample, initialRwv});

      FilteredClassifier initialClassifier = new FilteredClassifier();
      initialClassifier.setFilter(multiFilter);
      initialClassifier.setClassifier(AbstractClassifier.makeCopy(m_Classifier));

      Evaluation eval = new Evaluation(data);

      if (data.numInstances() > 2) {
        eval.crossValidateModel(initialClassifier, data, 2, rand);

        double[][] confusionMatrix = eval.confusionMatrix();

        for (int i = 2; i < indices.length; i++) {
          int index = indices[i];

          // Add one to avoid dividing by zero. Should not affect the output
          double p1 = confusionMatrix[index][initialIndices[0]] + 1;
          double p2 = confusionMatrix[index][initialIndices[1]] + 1;

          if (p1 / p2 > 1.0) {
            set1.add(index);
          } else {
            set2.add(index);
          }
        }
      } else {
        // If there's less than 2 instances, then just assign the classes randomly.
        // This should happen very rarely in practice.
        for (int i = 2; i < indices.length; i++) {
          if (i % 2 == 0) {
            set1.add(indices[i]);
          } else {
            set2.add(indices[i]);
          }
        }
      }
    }

    // Pick the classes for the current split
    int first = set1.size();
    int second = set2.size();
    int[] firstInds = collectionToIntArray(set1);
    int[] secondInds = collectionToIntArray(set2);

    // Sort the indices (important for hash key)!
    int[] sortedFirst = Utils.sort(firstInds);
    int[] sortedSecond = Utils.sort(secondInds);
    int[] firstCopy = new int[first];
    int[] secondCopy = new int[second];
    for (int i = 0; i < sortedFirst.length; i++) {
      firstCopy[i] = firstInds[sortedFirst[i]];
    }
    firstInds = firstCopy;
    for (int i = 0; i < sortedSecond.length; i++) {
      secondCopy[i] = secondInds[sortedSecond[i]];
    }
    secondInds = secondCopy;

    // Unify indices to improve hashing
    if (firstInds[0] > secondInds[0]) {
      int[] help = secondInds;
      secondInds = firstInds;
      firstInds = help;
      int help2 = second;
      second = first;
      first = help2;
    }

    m_Range = new Range(Range.indicesToRangeList(firstInds));
    m_Range.setUpper(data.numClasses() - 1);

    Range secondRange = new Range(Range.indicesToRangeList(secondInds));
    secondRange.setUpper(data.numClasses() - 1);

    // Change the class labels and build the classifier
    MakeIndicator filter = new MakeIndicator();
    filter.setAttributeIndex("" + (data.classIndex() + 1));
    filter.setValueIndices(m_Range.getRanges());
    filter.setNumeric(false);
    filter.setInputFormat(data);
    m_FilteredClassifier = new FilteredClassifier();
    if (data.numInstances() > 0) {
      m_FilteredClassifier.setClassifier(AbstractClassifier.makeCopies(
        classifier, 1)[0]);
    } else {
      m_FilteredClassifier.setClassifier(new weka.classifiers.rules.ZeroR());
    }
    m_FilteredClassifier.setFilter(filter);

    // Save reference to hash table at current node
    m_classifiers = table;

    if (!m_classifiers.containsKey(getString(firstInds) + "|"
      + getString(secondInds))) {
      m_FilteredClassifier.buildClassifier(data);
      m_classifiers.put(getString(firstInds) + "|" + getString(secondInds),
        m_FilteredClassifier);
    } else {
      m_FilteredClassifier = (FilteredClassifier) m_classifiers
        .get(getString(firstInds) + "|" + getString(secondInds));
    }

    // Create two successors if necessary
    m_FirstSuccessor = new RandomPairND();
    if (first == 1) {
      m_FirstSuccessor.m_Range = m_Range;
    } else {
      RemoveWithValues rwv = new RemoveWithValues();
      rwv.setInvertSelection(true);
      rwv.setNominalIndices(m_Range.getRanges());
      rwv.setAttributeIndex("" + (data.classIndex() + 1));
      rwv.setInputFormat(data);
      Instances firstSubset = Filter.useFilter(data, rwv);
      m_FirstSuccessor.generateClassifierForNode(firstSubset, m_Range, rand,
        classifier, m_classifiers);
    }
    m_SecondSuccessor = new RandomPairND();
    if (second == 1) {
      m_SecondSuccessor.m_Range = secondRange;
    } else {
      RemoveWithValues rwv = new RemoveWithValues();
      rwv.setInvertSelection(true);
      rwv.setNominalIndices(secondRange.getRanges());
      rwv.setAttributeIndex("" + (data.classIndex() + 1));
      rwv.setInputFormat(data);
      Instances secondSubset = Filter.useFilter(data, rwv);
      m_SecondSuccessor = new RandomPairND();

      m_SecondSuccessor.generateClassifierForNode(secondSubset, secondRange,
        rand, classifier, m_classifiers);
    }
  }

  public int[] collectionToIntArray(Collection<Integer> collection) {
    int[] ret = new int[collection.size()];
    int index = 0;
    Iterator<Integer> it = collection.iterator();

    while(it.hasNext()) {
      ret[index++] = it.next();
    }

    return ret;
  }

  /**
   * Returns default capabilities of the classifier.
   * 
   * @return the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    // class
    result.disableAllClasses();
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    // instances
    result.setMinimumNumberInstances(1);

    return result;
  }

  /**
   * Builds tree recursively.
   * 
   * @param data contains the (multi-class) instances
   * @throws Exception if the building fails
   */
  @Override
  public void buildClassifier(Instances data) throws Exception {

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();

    Random random = data.getRandomNumberGenerator(m_Seed);

    if (!m_hashtablegiven) {
      m_classifiers = new Hashtable<String, Classifier>();
    }

    // Check which classes are present in the
    // data and construct initial list of classes
    boolean[] present = new boolean[data.numClasses()];
    for (int i = 0; i < data.numInstances(); i++) {
      present[(int) data.instance(i).classValue()] = true;
    }
    StringBuffer list = new StringBuffer();
    for (int i = 0; i < present.length; i++) {
      if (present[i]) {
        if (list.length() > 0) {
          list.append(",");
        }
        list.append(i + 1);
      }
    }

    Range newRange = new Range(list.toString());
    newRange.setUpper(data.numClasses() - 1);

    generateClassifierForNode(data, newRange, random, m_Classifier,
      m_classifiers);
  }

  /**
   * Predicts the class distribution for a given instance
   * 
   * @param inst the (multi-class) instance to be classified
   * @return the class distribution
   * @throws Exception if computing fails
   */
  @Override
  public double[] distributionForInstance(Instance inst) throws Exception {

    double[] newDist = new double[inst.numClasses()];
    if (m_FirstSuccessor == null) {
      for (int i = 0; i < inst.numClasses(); i++) {
        if (m_Range.isInRange(i)) {
          newDist[i] = 1;
        }
      }
      return newDist;
    } else {
      double[] firstDist = m_FirstSuccessor.distributionForInstance(inst);
      double[] secondDist = m_SecondSuccessor.distributionForInstance(inst);
      double[] dist = m_FilteredClassifier.distributionForInstance(inst);
      for (int i = 0; i < inst.numClasses(); i++) {
        if ((firstDist[i] > 0) && (secondDist[i] > 0)) {
          System.err.println("Panik!!");
        }
        if (m_Range.isInRange(i)) {
          newDist[i] = dist[1] * firstDist[i];
        } else {
          newDist[i] = dist[0] * secondDist[i];
        }
      }
      return newDist;
    }
  }

  /**
   * Returns the list of indices as a string.
   * 
   * @param indices the indices to return as string
   * @return the indices as string
   */
  public String getString(int[] indices) {

    StringBuffer string = new StringBuffer();
    for (int i = 0; i < indices.length; i++) {
      if (i > 0) {
        string.append(',');
      }
      string.append(indices[i]);
    }
    return string.toString();
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {

    Vector<Option> newVector = new Vector<Option>(3);

    newVector.addElement(new Option(
            "\tPercentage of instances to be used in the \n"
          + "\ttraining of the initial classifier (default 100)",
            "P", 1, "-P"));
    newVector.addElement(new Option(
            "\tNumber of random pairs to choose, selecting\n"
          + "\tbest one for the actual split (default 1)",
            "N", 1, "-N"));

    newVector.addAll(Collections.list(super.listOptions()));

    return newVector.elements();
  }

  @Override
  public void setOptions(String[] options) throws Exception {

    String numRandomPairs = Utils.getOption('N', options);
    if (numRandomPairs.length() != 0) {
      setNumRandomPairs(Integer.parseInt(numRandomPairs));
    } else {
      setNumRandomPairs(1);
    }

    String subsamplePercent = Utils.getOption('P', options);
    if (subsamplePercent.length() != 0) {
      setSubsamplePercent(Double.parseDouble(subsamplePercent));
    } else {
      setSubsamplePercent(100.0);
    }

    super.setOptions(options);

    Utils.checkForRemainingOptions(options);
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String [] getOptions() {

    Vector<String> options = new Vector<String>();

    options.add("-P");
    options.add("" + getSubsamplePercent());

    options.add("-N");
    options.add("" + getNumRandomPairs());

    Collections.addAll(options, super.getOptions());

    return options.toArray(new String[0]);
  }

  public void setNumRandomPairs(int numRandomPairs) {
    m_numRandomPairs = numRandomPairs;
  }

  public void setSubsamplePercent(double percent) {
    m_subsamplePercent = percent;
  }

  public int getNumRandomPairs() {
    return m_numRandomPairs;
  }

  public double getSubsamplePercent() {
    return m_subsamplePercent;
  }

  /**
   * @return a description of the classifier suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String globalInfo() {

    return "A meta classifier for handling multi-class datasets with 2-class "
      + "classifiers by building a nested dichotomy with random-pair selection "
      + "to select the class subsets." ;
  }

  /**
   * Outputs the classifier as a string.
   * 
   * @return a string representation of the classifier
   */
  @Override
  public String toString() {

    if (m_classifiers == null) {
      return "RandomPairND: No model built yet.";
    }
    StringBuffer text = new StringBuffer();
    text.append("RandomPairND");
    treeToString(text, 0);

    return text.toString();
  }

  /**
   * Returns string description of the tree.
   * 
   * @param text the buffer to add the node to
   * @param nn the node number
   * @return the next node number
   */
  private int treeToString(StringBuffer text, int nn) {

    nn++;
    text.append("\n\nNode number: " + nn + "\n\n");
    if (m_FilteredClassifier != null) {
      text.append(m_FilteredClassifier);
    } else {
      text.append("null");
    }
    if (m_FirstSuccessor != null) {
      nn = m_FirstSuccessor.treeToString(text, nn);
      nn = m_SecondSuccessor.treeToString(text, nn);
    }
    return nn;
  }

  /**
   * Returns the revision string.
   * 
   * @return the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 10342 $");
  }

  /**
   * Main method for testing this class.
   * 
   * @param argv the options
   */
  public static void main(String[] argv) {
    runClassifier(new RandomPairND(), argv);
  }
}
