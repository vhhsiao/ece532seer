clear all;
clc;
f = load ("/Volumes/HSIAO USB/SEER/11_30/thyroidSEER.mat");
cd '/Volumes/HSIAO USB';
addpath("./SMOTE/functions")
warning('off', 'all');

seer_data = f.thyroidSEER;
n = height(seer_data);
vars = seer_data.Properties.VariableNames;

array_from_value = @(val, arr) repmat([val], length(arr), 1);

%% Standardize missing values
seer_data = standardizeMissing(seer_data, "Blank(s)");
seer_data = standardizeMissing(seer_data, "Unknown");
seer_data = standardizeMissing(seer_data, "NA");
seer_data = standardizeMissing(seer_data, 999);

%% Select dataset
% Isolate cases of papillary or follicular variant thyroid cancer
histologic_codes = [8260,8340,8350,8330,8331,8332,8335,8290];
COD_categories = ["Thyroid", "Alive"];
exclude_age_groups = ["00 years", "01-04 years", "05-09 years", "10-14 years", "15-19 years"];

include_histo = ismember(seer_data.HistologicTypeICDO3, histologic_codes);
include_COD = ismember(string(seer_data.CODToSiteRecode), COD_categories);
include_age = ~ismember(string(seer_data.AgeRecodeWith1YearOlds), exclude_age_groups);
seer_data = seer_data(include_histo & include_COD & include_age,:);

%% Recode variables of interest
seer_data.age = str2double(extractBetween(string(seer_data.AgeRecodeWith1YearOlds), 1, 2));
seer_data.sex = seer_data.Sex == "Female";

% TODO: finer granularity with T/N/M stages (1a, 1b etc)
seer_data.t = str2double(extractBetween(string(seer_data.DerivedAJCCT7thEd20102015), 2, 2));
seer_data.n = str2double(extractBetween(string(seer_data.DerivedAJCCN7thEd20102015), 2, 2));
seer_data.m = str2double(extractBetween(string(seer_data.DerivedAJCCM7thEd20102015), 2, 2));

% Regional nodes
regional_nodes_unavail = ismember(seer_data.RegionalNodesPositive1988, [97,98,99]);
seer_data.regional_nodes = seer_data.RegionalNodesPositive1988;
seer_data.regional_nodes(regional_nodes_unavail) = NaN;

% Extent
contained_in_thyroid = seer_data.CSExtension20042015 < 500;
minimal_spread = seer_data.CSExtension20042015 >= 500 & seer_data.CSExtension20042015 < 560;
gross_extrathyroidal_spread = seer_data.CSExtension20042015 >= 560;
seer_data.extent = seer_data.CSExtension20042015;
seer_data.extent(contained_in_thyroid) = 0;
seer_data.extent(minimal_spread) = 1;
seer_data.extent(gross_extrathyroidal_spread) = 2;

% Size
seer_data.size = seer_data.CSTumorSize20042015;
seer_data.size(seer_data.CSTumorSize20042015 == 991) = 5; % less than 1 cm/T1a
seer_data.size(seer_data.CSTumorSize20042015 == 992) = 15; % less than 2 cm, greater than 1cm, between 1cm and 2 cm
seer_data.size(seer_data.CSTumorSize20042015 == 993) = 25; % less than 3 cm, greater than 2cm, between 2cm and 3 cm
seer_data.size(seer_data.CSTumorSize20042015 == 994) = 35; % less than 4 cm, greater than 5cm, between 4cm and 5 cm, T2
seer_data.size(seer_data.CSTumorSize20042015 == 995) = 45; % less than 5 cm, greater than 4cm, between 4cm and 5 cm
seer_data.size(seer_data.CSTumorSize20042015 == 996) = 55; % greater than 5cm
seer_data.size(seer_data.CSTumorSize20042015 == 990) = 0; % microsoft focus/foci only and no size of focus given.

% One-hot encoding of race
seer_data.race_white = seer_data.RaceRecodeWBAIAPI == "White";
seer_data.race_black = seer_data.RaceRecodeWBAIAPI == "Black";
seer_data.race_api = seer_data.RaceRecodeWBAIAPI == "Asian or Pacific Islander";
seer_data.race_asian = seer_data.RaceRecodeWBAIAPI == "American Indian/Alaska Native";

%% Generate outcome variable
seer_data.ten_year_mortality = seer_data.CODToSiteRecode == "Thyroid" & seer_data.SurvivalMonths < 120;
seer_data.five_year_mortality = seer_data.CODToSiteRecode == "Thyroid" & seer_data.SurvivalMonths < 60;

%% Impute missing data values
input_vars = ["age", "sex", "t", "n", "m", "regional_nodes", "extent", "size", "race_white", "race_black", "race_api", "race_asian"];
seer_input = seer_data{:,input_vars};
seer_input = [ones(size(seer_input,1),1), seer_input];
seer_ten_year_mortality = seer_data.ten_year_mortality;

outcome = seer_data.ten_year_mortality;
oucome_svm = outcome;
outcome_svm(outcome == 0) = -1;
outcome_svm(outcome == 1) = 1;

save('seer_data', 'seer_input', 'seer_ten_year_mortality')

%% Data Preprocessing
% median impute
n = size(seer_input, 1);
%seer_input = fillmissing(seer_input, 'movmedian', n);

%% Split into train/test sets
[test_idx, train_idx] = crossvalind('Holdout', (size(seer_input,1)), 0.3);
seer_input_train = seer_input(train_idx,:);
seer_outcome_train = outcome(train_idx);
seer_input_test = seer_input(test_idx,:);
seer_outcome_test = outcome(test_idx);

%% SVM
fprintf("SVM")
seer_outcome_train_svm = outcome_svm(train_idx);
%mdl_svm = fitclinear(seer_input_train, seer_outcome_train_svm, 'Learner', 'svm', 'KFold', 10);
[mdl_svm,FitInfo,HyperparameterOptimizationResults] = fitckernel(seer_input_train,seer_outcome_train_svm,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));
[pred, score] = predict(mdl_svm, seer_input_train);
[fpr, tpr, t, train_auc] = perfcurve(seer_outcome_train, score(:,2), 1);
train_auc

[pred, score] = predict(mdl_svm, seer_input_test);
[fpr, tpr, t, test_auc] = perfcurve(seer_outcome_test, score(:,2), 1);
test_auc

%% ANN

