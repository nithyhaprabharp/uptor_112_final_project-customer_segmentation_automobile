## Customer Segmentation for Automobile Market Expansion

### Project Description
An automobile company is planning to expand into new markets with its existing product portfolio (P1, P2, P3, P4, and P5). Based on intensive market research, the company has determined that customer behavior in the new market closely mirrors that of its current market.  

In the existing market, the sales team successfully classified customers into four distinct segments (A, B, C, D) and tailored outreach strategies for each segment. This segmentation-based communication approach proved highly effective, driving customer engagement and product adoption.  

For the new market, the company has identified 3000 potential customers and intends to replicate the same segmentation strategy. By applying data-driven customer classification, the goal is to group these new customers into meaningful segments and design targeted marketing campaigns that align with their preferences and behaviors.

---

### Project Objective
To identify the most reliable machine learning model to classify new customers into the correct group, ensuring accurate segmentation for business decision-making. The model is trained and tested using existing market data, and then applied to predict the correct group (A, B, C, or D) for the 3000 new customers in the new market.

---

### Models Trained

The following classification models were trained and evaluated, including ensemble methods based on **Bagging** and **Boosting**:

- **Decision Tree**  
- **Random Forest** (Bagging)  
- **Bagging Classifier** (Bagging)  
- **AdaBoost** (Boosting)  
- **Gradient Boosting** (Boosting)  
- **XGBoost** (Boosting)
 

Each model was assessed using:  
- Accuracy  
- Weighted F1 Score  
- Macro F1 Score  
- Class-specific F1 Score (especially Class 3)  

---

### Bagging and Boosting 

- **Bagging (Bootstrap Aggregating):**  
  Bagging trains multiple models (often decision trees) on different random subsets of the training data and then aggregates their predictions.  
  - Strength: Reduces variance and helps prevent overfitting.  
  - Example: Random Forest and Bagging Classifier.  

- **Boosting:**  
  Boosting builds models sequentially, where each new model focuses on correcting the errors of the previous one.  
  - Strength: Reduces bias and improves accuracy by combining weak learners into a strong ensemble.  
  - Example: AdaBoost, Gradient Boosting, and XGBoost.  

In this project, we trained models with both **Bagging** and **Boosting** approaches to compare their effectiveness. Bagging methods provided stability and reduced variance, while Boosting methods delivered higher accuracy and stronger performance across critical customer segments.

---

### Best Model: Gradient Boosting

***Gradient Boosting*** emerged as the best overall model based on:
- Highest accuracy across all customer segments  
- Balanced F1 scores, ensuring fair classification  
- Strong performance for Class 3, the most critical segment for business impact  

This model significantly reduces misclassification risk and supports strategic decision-making.

---

### Final Report Summary

**Best Overall Model:** Gradient Boosting  

**Strengths:**
- High accuracy  
- Balanced performance across all segments  
- Reliable classification for high-value customers  

**Business Benefits:**
- Better targeting in marketing campaigns  
- Improved retention strategies  
- Fair and consistent classification  

---

### Conclusion
Based on our analysis, **Gradient Boosting** is the most effective model for predicting the right group of new customers. It delivers the highest accuracy and balanced performance across all customer segments, ensuring reliable classification.  

In this project, we trained models with **Bagging** (Random Forest, Bagging Classifier) and **Boosting** (AdaBoost, Gradient Boosting, XGBoost). While Bagging improved stability and reduced variance, Boosting methods consistently outperformed by correcting errors iteratively and achieving superior accuracy.  

The final Gradient Boosting model has been wrapped into a production-ready pipeline and deployed via a Flask API for real-time predictions. This enables managers to make data-driven decisions about customer targeting and engagement, ultimately improving business outcomes.


***Project Files***

- [ML_final_project_Automobile_Customer_segmentation_jupyter.ipynb](ML_final_project_Automobile_Customer_segmentation_jupyter.ipynb)
- [ML_final_project_Automobile_customer_segmentation_pycharm.py](ML_final_project_Automobile_customer_segmentation_pycharm.py)  
- [ML_final_project_Automobile_flask_api.py](ML_final_project_Automobile_flask_api.py)


***Data***

- [Train.csv](Train.csv)
- [Test.csv](Test.csv)
