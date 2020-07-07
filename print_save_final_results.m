clc
clearvars
close all

% old
% pat_list =  [559,563,570,575,588,591];

% new
pat_list =  [540,544,552,567,584,596];
% pat_list =  [584];

results_folder = 'test_final_pred'; % test_final_pred_onlyCGM | test_final_pred

for PH = [30 60]
    
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
            
            % convert string to numbers [comment if matlab version >= 2020]
            pred_data.prediction = my_str2numeric(pred_data.prediction);
            
            % shift prediction accounting for PH
            pred_data.prediction = shift_data(pred_data.prediction, PH/Ts);
            
            % realign prediction on original time grid
            yhat = realign_prediction(pred_data.prediction, pred_data.time, CGM_data.time);
            
            % calculate prediction metrics
            [rmse(repN), ~, ~, mae(repN), delay(repN)] = prediction_metrics(CGM_data.value, yhat, PH/Ts);
            % store every prediction
            pred_full = [pred_full yhat];
        end
        
        % take average of 5 prediction as final prediction ---------------
        CGM_data.prediction = mean(pred_full,2);
        CGM_data.prediction = round(CGM_data.prediction);
        
        % remove first hour
        CGM_data = CGM_data(12+1:end, :);
        
        % metrics
        [rmse_ensemble, ~, ~, mae_ensemble, delay_ensemble] = prediction_metrics(CGM_data.value, CGM_data.prediction, PH/Ts);
        
        % store results in table ---------------
        T_rmse{sprintf('%g', patN),{'RMSE_avg','RMSE_ensemble'}} = [mean(rmse), rmse_ensemble];
        T_results{sprintf('%g', patN),{'RMSE','MAE','delay', 'TG'}} = [rmse_ensemble, mae_ensemble, delay_ensemble*Ts, PH-delay_ensemble*Ts];
        
        all_results.(sprintf('PH%i',PH)).(sprintf('pat%i', patN)) = CGM_data;
    end
    
    T_results{'mean',:} = mean(T_results{:,:});
    ph_results.(sprintf('PH%i',PH)) = T_results;
    
    % print results
    disp(T_results)
    
end


%% make latex code for pdf
addpath('latexTable')

X = [ph_results.PH30{:,{'RMSE','MAE','TG'}} ph_results.PH60{:,{'RMSE','MAE','TG'}}];
T = array2table(X);
T = [table(T_results.Properties.RowNames) T];

% use this data
input.data = T;

% header
input.tableColLabels = {'ID','RMSE','MAE','TG','RMSE','MAE','TG'};
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

%% save on .txt
system_id = 'Cappon';

for PH = [30 60]
    my_results = all_results.(sprintf('PH%i',PH));
    for patN = pat_list
        T = my_results.(sprintf('pat%i', patN));
        T.value = [];
        save_name = sprintf('%s_%i_%i.txt',system_id,patN,PH);
        writetable(T, fullfile('results',results_folder,save_name),'Delimiter',' ','WriteRowNames',0);
    end
end

%% plot prediction
close all
figh = [];
ct = 1;

for PH = [30]% 60]
    for patN = 544%pat_list
        CGM_data = all_results.(sprintf('PH%i',PH)).(sprintf('pat%i', patN));
        
        % load training data ---------------
        fname = sprintf('Testing-%g-ws-testing.mat',patN);
        original_data = load(fullfile('data','original','Testing',fname));
        
        lh = [];
        
        figh(ct) = figure('Color','w');
        ax(1) = subplot(2,1,1);
        hold on
        lh(1) = plot(CGM_data.time, CGM_data.value, '-',...
            'LineWidth',2,'DisplayName', 'y');
        lh(2) = plot(CGM_data.time, CGM_data.prediction, '-',...
            'LineWidth',2,'DisplayName', '$\hat{y}$');
        ax(2) = subplot(2,1,2);
        stem(original_data.patient.timeseries.insulin_bolus.time_begin, ...
            original_data.patient.timeseries.insulin_bolus.value, '-',...
            'LineWidth',1)
        
        original_data.patient.timeseries.insulin_bolus
        
        my_legend = legend(lh, 'FontSize',12);
        set(my_legend, 'Interpreter', 'latex')
        ylabel('CGM [mg/dL]')
        xlabel('time')
        linkaxes(ax, 'x')
        
        ct = ct+1;
    end
end

% figure in paper
addpath('export_fig')
% fig = figure(figh(2));
% set(fig, 'Units','normalized')
xlim([datetime('30-Jun-2027 00:00:00') datetime('1-Jul-2027 12:00:00')])
% export_fig(fig, 'plot_example','-png')
set(findobj(gcf,'type','axis'), 'FontSize',12)

rmpath('export_fig')

%% count nan

T_nan_count.PH30 = table();
T_nan_count.PH60 = table();

for PH = [30 60]
    T = table();
    for patN = pat_list
        CGM_data_tmp = all_results.(sprintf('PH%i',PH)).(sprintf('pat%i', patN));
        CGM_data_tmp = CGM_data_tmp(~isnan(CGM_data_tmp.value),:);
        
        n_data = height(CGM_data_tmp);
        n_pred = sum(~isnan(CGM_data_tmp.prediction));
        pct_predicted = n_pred/n_data*100;
        
        T{sprintf('%g', patN),{'Ndata','Npred','pct'}} = [n_data, n_pred, pct_predicted];
    end
    T_nan_count.(sprintf('PH%i',PH)) = T;
end

T_nan_count.PH30
T_nan_count.PH60


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






