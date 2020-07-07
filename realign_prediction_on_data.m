clc
clearvars
close all

PH = 60;

% old
% pat_list =  [559,563,570,575,588,591];

% new
pat_list =  [540,544,552,567,584,596];
% pat_list =  [596];

results_folder = 'test_final_pred'; % test_final_pred_onlyCGM | test_final_pred

Ts = 5;
T_rmse = table();
T_results = table();

for patN = pat_list
    
    % load training data ---------------
    fname = sprintf('Testing-%g-ws-testing.mat',patN);
    original_data = load(fullfile('data','original','Testing',fname));
    CGM_data = original_data.patient.timeseries.CGM;
    
    % make every field of the structure a column array ---------------
    fields = fieldnames(CGM_data);
    for k = 1:length(fields)
        f = fields{k};
        CGM_data.(f) = CGM_data.(f)(:);
    end
    % make a table
    CGM_data = struct2table(CGM_data);
    
    % load prediction for every repetition ---------------
    pred_full = [];
    for repN = 1:5
        % load prediction
        fname = sprintf('subj%g_PH%g_test_rep%g.csv',patN,PH,repN);
        pred_data = readtable(fullfile('results',results_folder,fname));
        % convert string to numbers
        pred_data.prediction = my_str2numeric(pred_data.prediction);
        % shift prediction accounting for PH
        pred_data.prediction = shift_data(pred_data.prediction, PH/Ts);
        % realign prediction on original time grid
        yhat = realign_prediction(pred_data.prediction, pred_data.time, CGM_data.time);
        
        % post-process prediction using saturation on 1-step increases
%         yhat = saturate_increase(yhat, 30);

        % calculate prediction metrics
        [rmse(repN), ~, ~, mae(repN), delay(repN)] = prediction_metrics(CGM_data.value, yhat, PH/Ts);
        % store every prediction
        pred_full = [pred_full yhat];
    end
    
    % take average of 5 prediction as final prediction ---------------
    CGM_data.prediction = mean(pred_full,2);
    CGM_data.prediction = round(CGM_data.prediction);
    pred_error = abs(CGM_data.value - CGM_data.prediction);
    [rmse_ensemble, ~, ~, mae_ensemble, delay_ensemble] = prediction_metrics(CGM_data.value, CGM_data.prediction, PH/Ts);
    
    % store results in table ---------------
    T_rmse{sprintf('%g', patN),{'RMSE_avg','RMSE_ensemble'}} = [mean(rmse), rmse_ensemble];
    T_results{sprintf('%g', patN),{'RMSE','MAE','delay'}} = [rmse_ensemble, mae_ensemble, delay_ensemble*Ts];
    
    all_results.(sprintf('pat%i', patN)) = CGM_data;
end

T_results{'mean',:} = mean(T_results{:,:});

% print results
% disp(T_rmse)
disp(T_results)

%% save on .txt
system_id = 'Cappon';

for patN = pat_list
    T = all_results.(sprintf('pat%i', patN));
    T.value = [];
    
    save_name = sprintf('%s_%i_%i.txt',system_id,patN,PH);
    writetable(T, fullfile('results',results_folder,save_name),'Delimiter',' ','WriteRowNames',0);
end

%% save on pdf
addpath('latexTable')

T = T_results;
T.delay = [];
T.ID = T_results.Properties.RowNames;
T = T(:,{'ID','RMSE','MAE'});

% use this data
input.data = T;

% header
input.tableColLabels = {'ID','RMSE','MAE'};
% label and captions
input.tableLabel = 'performance';
input.tableCaption = 'Test-set performance';
% table settings
input.dataFormat = {'%.2f'};
input.tablePlacement = 'htbp';
input.tableColumnAlignment = 'c';
input.tableBorders = 1;
input.booktabs = 1;
input.makeCompleteLatexDocument = 1;
% make latex code
latex_results = latexTable(input);
rmpath('latexTable')

%% plot prediction
close all

figh = [];
ct = 1;

for patN = pat_list
    CGM_data = all_results.(sprintf('pat%i', patN));
    
    figh(ct) = figure('Color','w');
    hold on
    plot(CGM_data.time, CGM_data.value, 'DisplayName', 'y')
    plot(CGM_data.time, CGM_data.prediction, 'DisplayName', '$\hat{y}$')
    my_legend = legend('FontSize',12);
    set(my_legend, 'Interpreter', 'latex')
    ylabel('CGM [mg/dL]')
    xlabel('time')
    
    ct = ct+1;
end

%%
addpath('export_fig')
fig = figure(figh(2));
set(fig, 'Units','normalized')
xlim([datetime('30-Jun-2027 00:00:00') datetime('1-Jul-2027 12:00:00')])
export_fig(fig, 'plot_example','-pdf','-png')
set(gca, 'FontSize',12)

rmpath('export_fig')

%%
function ynew = shift_data(y, n)
% Shift y forward by n samples, replace with nan at the beginning
ynew = [nan*ones(n,1); y];
ynew = ynew(1:length(y));
end

%%
function yhat_new = realign_prediction(yhat_vals, yhat_time, y_time)
% Realign prediction values on original time grid

yhat_new = zeros(size(y_time));
for n = 1:length(y_time)
    % original time
    t = y_time(n);
    % find closest time in predicted values
    time_distance = t - yhat_time;
    
    % take only past values
    %     time_distance(time_distance < 0) = nan;
    %     [~,b] = min(time_distance);
    
    % take nearest values
    [~,b] = min(abs(time_distance));
    
    % store predicted value
    yhat_new(n) = yhat_vals(b);
end

end

%%
function ynew = my_str2numeric(y)
% Convert a table column of strings into numeric format, place nan if empty

ynew = zeros(size(y));
for k = 1:length(y)
    x = y(k);
    x = x{1};
    x = str2double(x);
    if isempty(x)
        x = nan;
    end
    ynew(k) = x;
end

end

%%
function y_new = saturate_increase(y, max_val)
%

increase = [0; diff(y)];

increase2 = increase;
increase2(increase2 > max_val) = max_val;
increase2(increase2 < -max_val) = -max_val;

post_processing_adj = increase - increase2;

y_new = y - post_processing_adj;

end






