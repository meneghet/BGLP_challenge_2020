function [rmse, cod, fit, mae, delay] = prediction_metrics(y, yhat, PH)

% quadratic error
pred_error = y - yhat;
quad_err = pred_error.^2;

% rmse
rmse = sqrt(nanmean(quad_err));

% cod
SSE = nansum(quad_err);
var_y = (y - nanmean(y)).^2;
SST = nansum(var_y);
cod = 100*(1-(SSE/SST));

% fit
fit = 100*(1-sqrt(SSE/SST));

% mean absolute error
mae = nanmean(abs(pred_error));

% calculate delay
N = length(y);
for j = 0:PH
    for i = 1:N-PH-j
        indx = i+PH+j;
        dif(i) = (yhat(indx)-y(i+PH))^2;
    end
    count = nansum(dif);
    delay_tmp(j+1) = (1/(N-PH+1))*count;
end
[~,d] = min(delay_tmp);
delay = d-1; % because we used j+1 as index


end