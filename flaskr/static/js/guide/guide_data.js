var tip_home =  
[
    {
        element: '#upload_icon',
        tooltip: 'Upload your pkl/csv files to pre-process !',
        text: 'Upload File',
        files: [{url: '', name: 'more details'}],
        position: 'T'
    },
    {
        element: '#visualization_icon',
        tooltip: 'Visualize the patterns of your data. Explore in detail the box plots, histograms, scatter plots and many more !',
        text: 'Visualization',
        position: 'T'
    },
    {
        element: '#preprocess_icon',
        tooltip: 'Pre-process the raw data set. Map into gene symbols, normalize, handle null values and filter deferentially expressed genes based on p-values and fold change !',
        text: 'Pre-processing',
        position: 'T'
    },
    {
        element: '#feature_icon',
        tooltip: 'Select the number of features you want and the methods to use. Examine the set of selected features and the accuracy of selected features by each method !',
        text: 'Feature Selection',
        position: 'T'
    },
    {
        element: '#analize_icon',
        tooltip: 'Use this feature to dive deeper into the selected features. Analyze the selected set of features with their accuracies, correlation values, and the number of features !',
        text: 'Analysis',
        position: 'T'
    },
    {
        element: '#validation_icon',
        tooltip: 'Validate the results obtained by referring to the already published/ accepted data sources !',
        text: 'Validation',
        position: 'T'
    },
    {
        element: '#modeling_icon',
        tooltip: 'Develop a model for a data set by using the previously extracted feature set. Select the classifier you want to generate the model !',
        text: 'Modelling',
        position: 'T'
    },
    {
        element: '#prediction_icon',
        tooltip: 'Predict the class of a newly provided sample/samples based on the previous data !',
        text: 'Prediction',
        position: 'T'
    },
    {
        element: '.notification_pop',
        tooltip: 'Tour pop-up disable or enable !',
        text: 'Tour notification',
        position: 'L'
    }
];

var tip_step_1 =  
[
    { 
        element : '#available_file_name', 
        tooltip : 'Choose a file from the list !', 
        text: 'Availabe files', 
        files: [{ url : '', name : 'more details' }], position: 'T' 
    },
    { 
        element : '#view_div', 
        tooltip : 'You can view the selected data set !', 
        text: 'Show Data set', 
        position: 'T' 
    },
    { 
        element : '#annotation_table_name', 
        tooltip : 'Choose the annotation table corresponding to the chosen data set to map probe ids to gene symbols !', 
        text: 'Annotation table', 
        position: 'T' 
    },
    { 
        element : '#average_div', 
        tooltip : 'Choose a method to handle multiple probe ids mapping to the same gene symbol !', 
        text: 'Probe selection method', 
        position: 'T' 
    }
];

var tip_step_2 =
[
    { 
        element : '#number_features_div', 
        tooltip : 'Number of probe ids in the data set !', 
        text: 'Number of features', 
        position: 'T' 
    },
    { 
        element : '#number_samples_div', 
        tooltip : 'Number of samples in the data set !', 
        text: 'Number of samples', 
        position: 'T' 
    },
    { 
        element : '#number_samples_positive', 
        tooltip : 'Class positive samples in the data set !', 
        text: 'Class positive samples', 
        position: 'T' 
    },    
    { 
        element : '#min_val_div', 
        tooltip : 'Minimum gene expression data value in all the samples (considering all the features) !', 
        text: 'Minimum value', 
        position: 'T' 
    },
    { 
        element : '#max_val_div', 
        tooltip : 'Maximum gene expression data value in all the samples (considering all the features) !', 
        text: 'Maximum value', 
        position: 'T' 
    },
    { 
        element : '#number_samples_negative', 
        tooltip : 'Class negative samples in the data set !', 
        text: 'Class negative samples', 
        position: 'T' 
    },
    { 
        element : '#table_div', 
        tooltip : 'You can view a portion of the data set. Scroll right to view more samples !', 
        text: 'Show Data set', 
        position: 'T' 
    }    
];

var tip_step_3 =
[
    { 
        element : '#number_features_div', 
        tooltip : 'Number of probe ids in the data set !', 
        text: 'Number of features', 
        position: 'T' 
    },
    { 
        element : '#number_null_div', 
        tooltip : 'A count of all the occurrences where the gene expression value  is missing !', 
        text: 'Number of null values', 
        position: 'T' 
    },
    { 
        element : '#number_unique_div', 
        tooltip : 'The number of distinct gene symbols after the mapping of probe ids to gene symbols is performed !', 
        text: 'Number of unique Gene Symbols', 
        position: 'T' 
    },
    { 
        element : '#number_gene_null_div', 
        tooltip : 'The number of probe ids where a corresponding gene symbol is not found in the annotation table !', 
        text: 'Number of null values in Gene Symbols', 
        position: 'T' 
    },
    { 
        element : '#min_val_div', 
        tooltip : 'Minimum gene expression data value in all the samples (considering all the features) !', 
        text: 'Minimum value', 
        position: 'T' 
    },
    { 
        element : '#max_val_div', 
        tooltip : 'Maximum gene expression data value in all the samples (considering all the features) !', 
        text: 'Maximum value', 
        position: 'T' 
    },
    { 
        element : '#scaling_select_div', 
        tooltip : 'Scale the data to a specific range by selecting an appropriate method !', 
        text: 'Scaling', 
        position: 'T' 
    },
    { 
        element : '#imputation_select_div', 
        tooltip : 'Null values should be handled properly before moving to next steps. Use these techniques to handle null values !', 
        text: 'Imputation', 
        position: 'T' 
    }
];

var tip_step_5 =
[
    { 
        element : '#img_results_div', 
        tooltip : 'Volcano Plot :Fold change vs -log10 (p-value) !', 
        text: 'Fold vs p-value plot', 
        position: 'R' 
    },
    { 
        element : '.use-download-fold', 
        tooltip : 'Download this graph !', 
        text: 'Download image', 
        position: 'T' 
    },
    { 
        element : '#fold_range_div', 
        tooltip : 'Fold change is the diﬀerence between the means of the two groups of samples. You can select a value to filter the genes that have a greater fold value !', 
        text: 'Fold Range', 
        position: 'L' 
    },
    { 
        element : '#p_value_range_div', 
        tooltip : 'The p-value is a measure of how likely you are to get this spot data if no real difference existed. Therefore, a small p-value indicates that there is a small chance of getting this data if no real difference existed and therefore you decide that the difference in group expression data is significant.  Usually, we select 0.05 as the cut off for p-values!', text: 'P-value', position: 'B' },
    { 
        element : '#calculate_feature_div', 
        tooltip : 'By selecting cut off values for p-values and the fold change, you can filter out deferentially expressed genes !', 
        text: 'Filter out feature', 
        position: 'T' 
    }
];

var tip_step_6 =
[
    { 
        element : '#img_results_div', 
        tooltip : 'Uni-variate selection ranks the genes individually, from the most to the least informative. By examining the knee-point of this graph, we can see the number of genes that shows  significant difference !', 
        text: 'Uni-variate selection', 
        position: 'R' 
    },
    { 
        element : '.use-download-univarient_img', 
        tooltip : 'Download this graph !', 
        text: 'Download image', 
        position: 'T' 
    },
    { element : '#table_results_div', 
        tooltip : 'Classification is the process of predicting the class of given data points. A classifier is a function that is used to assign the samples to a class. Classification accuracy of the selected genes using 5 classifiers is shown in the table !', 
        text: 'Classification results', 
        position: 'L' 
    },
    { 
        element : '#highest_accuracy_clf', 
        tooltip : 'The testing and training accuracy of the filtered features with 6 classifiers are obtained previously. Out of these, we will select the best 3 classifiers to proceed in  the next steps !', 
        text: 'Highest accurate classifiers', 
        position: 'L' 
    },
    { 
        element : '#select_num_features_div', 
        tooltip : 'The number of features can be reduced by examining the knee-point of the uni-variate selection graph. Select the feature count to proceed to next steps !', 
        text: 'Select feature count', 
        position: 'T' 
    }
];

var tip_fs_index =
[
    { 
        element : '#selected_file_div', 
        tooltip : 'This is the file you will have after going through previous steps. It will be automatically selected. You do not need to change anything, if you wish to proceed with the same file !', 
        text: 'Selected file', 
        position: 'T' 
    },
    { 
        element : '#change_file_div', 
        tooltip : 'You can also perform feature selection on a separate file without going through the pre-processing steps. Use this feature if you want to select a new file !', 
        text: 'Change file', 
        position: 'T' 
    },
    { 
        element : '#select_methods_div', 
        tooltip : 'Feature selection is the process of selecting the features that contribute mostly to the target variable.  Different feature selection methods are used for this purpose. In this application, you may select 3 feature selection methods to get the best set of features !', 
        text: 'Feature Selection Methods', 
        position: 'T' 
    },
    { 
        element : '#num_of_features', 
        tooltip : 'Specify the number of features you want to filter out by the above feature selection methods !', 
        text: 'Number of Features Selected', 
        position: 'T' 
    }
];

var tip_fs_results =
[
{ 
    element : '#results_venn_div', 
    tooltip : 'This shows the Venn diagram of the number of genes found by each feature selection method. Click on different sections of the Venn diagram to view more information on the list of gene symbols !', 
    text: 'Venn diagram', 
    position: 'T' 
},
{ 
    element : '#results_img_div', 
    tooltip : 'These graphs depicts how the training and testing accuracy of the selected features from the previous step varies with different feature selection methods and the classifiers. Accuracy is given by the y- axis. 3 classifiers are used to give a comparison between the feature selection methods. Each feature selection method is performed and trained under all the 3 classifiers !', 
    text: 'Result plot', 
    position: 'T' 
},
{ 
    element : '#list_div_btn', 
    tooltip : 'Now you have got a subset of the best features. Click this if you want to view them as a list !', 
    text: 'List layout', 
    position: 'T' 
},
{ 
    element : '#grid_div_btn', 
    tooltip : 'Click this to view the features in a grid layout !', 
    text: 'Grid layout', 
    position: 'T' },
{ 
    element : '#results_gene_div', 
    tooltip : 'View all the gene symbols of the features selected by each feature selection method !', 
    text: 'Result gene list', 
    position: 'T'
}
];

var tip_an_index =
[
    { 
        element : '#result_plot_div1', 
        tooltip : 'Right: Number of features selected by each method only, Left : The variation of accuracy with the features selected by one method trained under 3 classifiers !', 
        text: 'Training accuracies', 
        position: 'T' 
    },
    { 
        element : '#result_plot_div2', 
        tooltip : 'Variation of accuracy of incrementally smaller sets of genes selected by each feature selection method !', 
        text: 'Classification accuracy', 
        position: 'T' 
    },
    { 
        element : '#result_plot_div3', 
        tooltip : 'Correlation matrix of the selected features that are only obtained through that method !', 
        text: 'Correlation matrix', 
        position: 'T' 
    },
    { 
        element : '#select_method_div', 
        tooltip : 'Select one of the feature selection method to proceed further. By doing this, you will be able to filter out the best set of genes given by the selected feature selection method !', 
        text: 'Select one methord', 
        position: 'T' 
    }
];

var tip_an_correlation =
[
    { 
        element : '#results_plot_div1', 
        tooltip : 'Variation of correlation values with the number of correlated features by considering all the 3 methods !', 
        text: 'Correlated features', 
        position: 'T' 
    },
    { 
        element : '#results_plot_div2', 
        tooltip : 'Variation of correlation coefficient with number of features and the variation of correlation coefficients with classification accuracy !', 
        text: 'Variation of the correlation coefficients', 
        position: 'T' 
    },
    { 
        element : '#results_table_div', 
        tooltip : 'Filtering out the least correlated features (to minimize redundancy) with the highest classification accuracy !', 
        text: 'Correlation matrix', 
        position: 'T' 
    },
    { 
        element : '#overlapped_features_div', 
        tooltip : 'You can view the gene set that was selected by all three feature selection methods. This is the overlap set of genes !', 
        text: 'Overlapped features', 
        position: 'T' 
    },
    { 
        element : '#correlation_features_div', 
        tooltip : 'You can view the gene set that was obtained by considering the correlation coefficients and the highest accuracy !', 
        text: 'Correlation | {{ max_clasify }}', 
        position: 'T' 
    }
];

var tip_an_final =
[
    { 
        element : '#gene_output_div1', 
        tooltip : 'This shows the set of genes selected by considering the least correlation values and the highest accuracy !', 
        text: 'Extracted genes', 
        position: 'T' 
    },
    { 
        element : '#results_plot_div1', 
        tooltip : 'ROC curves and the correlation matrix of the selected genes !', 
        text: 'ROC and Correlation matrix', 
        position: 'T' 
    },
    { 
        element : '#results_table_div1', 
        tooltip : 'This table shows the testing and training accuracies of the above selected genes !', 
        text: 'Accuracy results', 
        position: 'T' 
    },
    { 
        element : '#gene_output_div2', 
        tooltip : 'This shows the set of genes selected by considering the  union of overlapped genes and the set of genes found by considering correlation values !', 
        text: 'Extracted genes', 
        position: 'T' 
    },
    { 
        element : '#results_plot_div2', 
        tooltip : 'ROC curves and the correlation matrix of the selected genes !', 
        text: 'ROC and Correlation matrix', 
        position: 'T' 
    },
    { 
        element : '#results_table_div2', 
        tooltip : 'This table shows the testing and training accuracies of all the selected features. This is the final accuracy of the  best set of genes !', 
        text: 'Accuracy results', 
        position: 'T' 
    }
];

var tip_validation =
[
    { 
        element : '#info_results_div', 
        tooltip : 'Now, we need to validate our results to confirm their biological relevance. For this, we use data from https://www.genecards.org/ to compare our results. In GeneCards, they have information on the already identified risk genes for a particular disease!', 
        text: 'Validation', 
        position: 'T' 
    },
    { 
        element : '#venn_diagram_div', 
        tooltip : 'This shows the Venn diagram of the number of genes found by each feature selection method which are also validated by other biological sources. Click on different sections of the Venn diagram to view more information on the list of gene symbols !', 
        text: 'Venn diagram', 
        position: 'T' 
    },
    { 
        element : '#results_table_div1', 
        tooltip : 'This table gives information on the already identified genes !', 
        text: 'Validation', 
        position: 'T' 
    }
];

var tip_model_index =
[
    { 
        element : '#extracted_features_div', 
        tooltip : 'These are the final set of features that were extracted from the previous steps !', 
        text: 'Extracted features', 
        position: 'T' 
    },
    { 
        element : '#choose_file_div', 
        tooltip : 'Choose a final containing the training data to be used in modelling !', 
        text: 'Choose file ', 
        position: 'T' 
    },
    { 
        element : '#classifier_method_div', 
        tooltip : 'Select a classifier to proceed with developing the model !', 
        text: 'Classifier', 
        position: 'T' 
    }
];

var tip_model_predict =
[
    { 
        element : '#modal_create', 
        tooltip : 'The process of modeling means training a machine learning algorithm to predict the labels from the features. The output from modeling is a trained model that can be used for making predictions on new data sets !', text: 'Custmize model', position: 'T' },
    { 
        element : '#select_file_div', 
        tooltip : 'Choose the file that you want to predict the target output !', 
        text: 'Choose File ', 
        position: 'T' 
    },
    { 
        element : '#select_normalize_div', 
        tooltip : 'You can choose to perform the normalization on the selected data set too by turning this feature on !', 
        text: 'Normalize method', 
        position: 'T' 
    },
    { 
        element : '#select_annotation_div', 
        tooltip : 'You can map the probe ids to gene symbols by performing annotation table mapping. Turn on this feature to proceed with table mapping !', 
        text: 'Annotation mapping', 
        position: 'T' 
    }
];

