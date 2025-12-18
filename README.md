## A Sequential Minimal Optimisation SVM in R

* Uncompress data in inst/extdata/final_labeled_data.zip
* Prepare data with vignettes/data_analysis.qmd, which also contains code to
* configure and train a SmoModel.

### Training and Predicting

```
model_balanced <- SmoModel$new(x = X_train_balanced, y = y_train_balanced,
                               kernel = "gaussian", verbose = TRUE, C = 5,
                               sigma = 40.0, tol = 0.001)
fit_smo(model_balanced, max_iterations = 1000)
#saveRDS(model_balanced, file = "my_preferred_path")
#readRDS(model_balanced)
model_balanced

# Training
mb_train_predict <- model_balanced$predict(X_train_balanced)
mb_cm_train  <- confusionMatrix(as.factor(y_train_balanced),
                                as.factor(mb_train_predict),
                                positive = "1")
```
