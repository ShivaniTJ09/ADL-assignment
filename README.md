# ADL-assignment

Here is the answer the question 4 

# Faster LTN

## Dataset

  you can download data from here- http://roozbehm.info/pascal-parts/pascal-parts.html
- `pascalpart_dataset.tar.gz`: it contains the annotations (e.g., small specific parts are merged into bigger parts) of pascalpart dataset in pascalvoc style. This folder is necessary if you want to train Fast-RCNN (https://github.com/rbgirshick/fast-rcnn) on this dataset for computing the grounding/features vector of each bounding box.
   

## Axioms

Axioms are included to specify that a part cannot include another part, that a whole object cannot include another whole object, and that each whole is generally associated with a set of given parts.
A grounded theory that considers also mereological constraints as prior knowledge can be constructed by adding such axioms to 

![image](https://user-images.githubusercontent.com/85010143/147633243-7c7875f1-c3f1-421b-b505-c68b37b1f029.png )

defined this more formally 

  ![image](https://user-images.githubusercontent.com/85010143/147633326-86d653a7-bbd3-4916-a0e4-461915c98bdb.png)

1) The first containing only training examples of object types and partof relations (𝑇𝑒𝑥𝑝𝑙)

* Mutual exclusion constraints
  
  
 ![image](https://user-images.githubusercontent.com/85010143/147626308-890d5fdb-cfc8-4763-a636-de5bda371f1c.png)

Here they have used the Mutual exclusion axiom. we can desribe mutual exclusion is a property of concurrency control, which is instituted for the purpose of preventing race conditions.it can be described into K(K − 1))/2 clauses, corresponding to all unordered class pairs over K classes, e.g., Cat(x) ⇒ ¬Person(x)

The second containing also logical axioms about types and part-of (𝑇𝑝𝑟𝑖𝑜𝑟)

* Mererological Constraints

 ![image](https://user-images.githubusercontent.com/85010143/147626354-39c31223-51f2-43d7-aa87-fd5bd1d746bf.png)

this indicate that if an object y is classified as part of x and x is a cat, thany can be only an object that we know is a part of the whole cat. Mereological constraints were enforced exploiting the KB developement, to which the reader is referred for further information.



## Comparison of Faster R-CNN and Faster-LTN 

![image](https://user-images.githubusercontent.com/85010143/147636084-4e40b6e6-6b9f-4196-9907-dfe8a6b53cac.png)


## Other Ground Functions
 
 An example of groundings for predicates: can be defined by  taking a one-vs-all multi-classifier approach 
 1) Define the following grounding for each class » : the vector corresponding to the grounding of a bounding box

![image](https://user-images.githubusercontent.com/85010143/147656851-dd2bef2e-8b55-4265-99c7-333da75b07f1.png)


2) a simple rule-based approach for defining a grounding for the partOf relation is based on the naive assumption that » The more a bounding box 𝑏 is contained within a bounding box 𝑏’, the higher the probability should be that 𝑏 is part of 𝑏’» : of bounding box 𝑏, with grounding 𝒙, into bounding box b‘, with grounding 𝒙’

![image](https://user-images.githubusercontent.com/85010143/147657096-fade4824-3b41-4516-880e-0d16c12c4fe1.png)














 
 
 
