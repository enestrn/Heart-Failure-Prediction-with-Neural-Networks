data = readtable('heart_failure_clinical_records_dataset.csv');
summary(data);
X = data(:, {'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', ...
    'ejection_fraction', 'high_blood_pressure', 'platelets', ...
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking'});
X = normalize(X);
y = data.DEATH_EVENT;
cv = cvpartition(size(data, 1), 'HoldOut', 0.2);
train_idx = training(cv);
test_idx = test(cv);
X_train = X(train_idx, :);
y_train = y(train_idx);
X_test = X(test_idx, :);
y_test = y(test_idx);
X_train = table2array(X_train);
X_test = table2array(X_test);
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.trainFcn = 'trainlm';
net.performFcn = 'mse';
X_train = X_train';
y_train = y_train';
X_test = X_test';
y_test = y_test';

% Eğitim
[net, tr] = train(net, X_train, y_train);

% Test 
y_pred = net(X_test);

% Metriklerin performans hesaplaması
perf = perform(net, y_test, y_pred);
RMSE = sqrt(mean((y_pred - y_test).^2));
MAE = mean(abs(y_pred - y_test));

disp(['Model Performansı (MSE): ', num2str(perf)]);
disp(['Root Mean Squared Error (RMSE): ', num2str(RMSE)]);
disp(['Mean Absolute Error (MAE): ', num2str(MAE)]);

% Model performansı 
figure;
plotperform(tr);
figure;
plot(y_test, 'b');
hold on;
plot(y_pred, 'r');
legend('Gerçek Değerler', 'Tahmin Değerleri');
title('Tahmin ve Gerçek Değerlerin Karşılaştırılması');
xlabel('Örnekler');
ylabel('DEATH_EVENT');

net.layers{1}.transferFcn = 'tansig';
net.trainParam.lr = 0.01; 
net.trainParam.epochs = 100; 
net.divideFcn = 'dividerand'; 
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;
net.trainParam.max_fail = 6;
% Yeniden eğitim
[net, tr] = train(net, X_train, y_train);
% Retest 
y_pred = net(X_test);
perf = perform(net, y_test, y_pred);
RMSE = sqrt(mean((y_pred - y_test).^2));
MAE = mean(abs(y_pred - y_test));

disp(['Optimizasyon Sonrası Model Performansı (MSE): ', num2str(perf)]);
disp(['Root Mean Squared Error (RMSE): ', num2str(RMSE)]);
disp(['Mean Absolute Error (MAE): ', num2str(MAE)]);

figure;
plot(y_test, 'b');
hold on;
plot(y_pred, 'r');
legend('Gerçek Değerler', 'Tahmin Değerleri');
title('Optimizasyon Sonrası Tahmin ve Gerçek Değerlerin Karşılaştırılması');
xlabel('Örnekler');
ylabel('DEATH_EVENT');

results = table({'MSE'; 'RMSE'; 'MAE'}, [perf; RMSE; MAE], ...
    'VariableNames', {'Metric', 'Value'});
disp(results);
