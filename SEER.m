clear all;
clc;
f = load ("/Volumes/HSIAO USB/SEER/11_30/SEER_thyroid.csv");
cd '/Volumes/HSIAO USB';
addpath("./SMOTE/functions")
warning('off', 'all');

seer_data = f.seer;
n = height(seer_data);
vars = seer_data.Properties.VariableNames;

array_from_value = @(val, arr) repmat([val], length(arr), 1);

% Converts all cell types to categorical
%seer_data = convertvars(seer_data, @iscell, "categorical");

% Convert default -99 values to standard missing value
%seer_data = standardizeMissing(seer_data, -99);

%categorical_vars = ["race_new", "cpt", "anesthes", "surgspec", "diabetes", "thy_indication", "thy_clintox", "thy_necksurg", "thy_needlebiop", "thy_oper_appr", "thy_neoplasm_type", "thy_tumor_t", "thy_multifocal", "thy_lymph_n", "thy_distantm_m", "thy_para", "dischdest", "wound_closure", "podiag10"];
categorical_vars = ["race_new"; "cpt"; "anesthes"; "surgspec"; "diabetes"; "thy_indication"; "thy_clintox"; "thy_necksurg"; "thy_needlebiop"];

%% Impute preoperative variables with some values missing
exclude_vars = [];
for i=1:length(preoperative_variables_all)
    v = preoperative_variables_all(i);
    v_class = class(seer_data.(char(v{1})));
    prop_missing = sum(ismissing(seer_data(:,v)))/n;
    %fprintf("%s", v{1});
    %fprintf(":   %.3f ", prop_missing);
    %fprintf("   type %s \r\n", v_class);
    if prop_missing > 0.2
        exclude_vars = [exclude_vars; v];
    end
    if v_class == "double"
        % TODO get knn imputation working
        %seer(:,v) = knnimpute(seer(:,v));
        nan = isnan(seer_data{:,v});
        seer_data{:,v}(nan) = ones(sum(nan),1) * median(seer_data{:,v}(~nan));
        fprintf("Numeric variable imputation by median: %s %d (%.3f)\n\n", v, size(nan,1), prop_missing);
    else
        % Sample empirically from the distribution
        categorical_vals = unique(seer_data{:,v});
        n_vals = size(categorical_vals,1);
        p = zeros(1,n_vals);
        for i=1:n_vals
            p(i) = sum(seer_data{:,v} == categorical_vals(i));
        end
        p = p ./ sum(p);
        missing = find(ismissing(seer_data{:,v}));
        for i=missing
            seer_data{i,v} = categorical_vals(mnrnd(1,p) == 1);
        end
        fprintf("Categorical variable imputation by sampling: %s %d (%.3f)\n\n", v, size(missing,1), prop_missing);
        for i=1:length(categorical_vals)
            val = string(categorical_vals(i));
            if strlength(v) + strlength(val) > 62
                max_val_length = 62 - strlength(v);
                shortened_name = extractBetween(val, 1, max_val_length);
                fprintf("Shortening categorical variable name %s -> %s\n", v, shortened_name);
                seer_data.(v)(seer_data.(v)==val) = categorical(array_from_value(shortened_name, seer_data.(v)(seer_data.(v)==val)));
            end
        end
    end
end

%% One-hot Encoding for categorical variables

%% Logistic Regression with ElasticNet Regularization

%% SVM

%% ANN