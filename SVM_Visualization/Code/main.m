clear;
clc;
close all;
%calling support vector machine function for handwritten digits
%classification
[y,d] = support_vector('X_train.csv','X_test.csv', 'T_train.csv', 'T_test.csv');