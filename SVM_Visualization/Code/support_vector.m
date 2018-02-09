function [x,y] = support_vector(train, test, trainlabel, testlabel)

% Reading the data given in csv files
trainData = xlsread(train);
trainLabel = xlsread(trainlabel);
testData = xlsread(test);
testLabel = xlsread(testlabel);

% addpath to the libsvm toolbox
addpath('C:/Users/Siffi Singh/Desktop/libsvm-3.22/libsvm-3.22/matlab');

% addpath to the data
dirData = 'C:/Users/Siffi Singh/Desktop/libsvm-3.22'; 
addpath(dirData);

% Grid search to find best_C best_gamma
% grid of parameters
folds = 5;
bestcv = 0;
for log2c = -1:1:1
  for log2g = -1:1:1
    cmd = ['-q -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
    cv = svmtrain(trainLabel,  [(1:5000)' trainData*trainData'], sprintf('-c %f -g %f -v %d', 2^log2c, 2^log2g, folds));
    if (cv >= bestcv)
      bestcv = cv; bestc = 2^log2c; bestg = 2^log2g; 
    end
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
  end
end

% (trainLabel, [(1:5000)' trainData*trainData'], '-c 1 -g 0.07 -b 1 -t 4');
model = svmtrain(trainLabel, [(1:5000)' trainData*trainData'], sprintf('-c %f -g %f -b 1 -t 4', best_C, best_gamma));
% Use the SVM model to classify the data
[predict_label, accuracy, prob_values] = svmpredict(testLabel, [(1:2500)' testData*trainData'], cv, '-b 1'); % test the training data

x = predict_label;

% PCA on testData for dimentionalty reduction
[coeff, score] = pca(testData);
X = score(:,1:2)

% Plotting the data before SVM, with given labels
[d,n] = size(X');
X = X';
testLabel = testLabel';
assert(n == length(testLabel));
color = 'brgmcyk';
m = length(color);
c = max(testLabel);

figure(1)
figure(gcf);
clf;
hold on;
view(2);
   for i = 1:c
       idc = testLabel==i;
       scatter(X(1,idc),X(2,idc),36,color(mod(i-1,m)+1), 'filled');
   end
       title('Original Data with 2D PCA');
       legend('Class 1', 'Class 2', 'Class 3','Class 4','Class 5');
%        decision(testData, testLabel);
axis equal
grid on
hold off

% Plotting the data after SVM, with predicted labels
predict_label = predict_label';
assert(n == length(predict_label));
c = max(predict_label);
figure(2)
figure(gcf);
clf;
hold on;
view(2);
   for i = 1:c
       idc = predict_label==i;
       scatter(X(1,idc),X(2,idc),36,color(mod(i-1,m)+1), 'filled');
   end
       title('Predicted Data with 2D PCA');
       legend('Class 1', 'Class 2', 'Class 3','Class 4','Class 5');        
axis equal
grid on
hold off

y = model;

%Plotting the decision boundary begins and support vectors
% Labels are 1, 2, 3, 4 or 5
groundTruth = testLabel;
d = X;

%Support vectors
sv = full(model.SVs);
sv_idx = full(model.sv_indices);
figure(3)
figure(gcf);
clf;
hold on;
% Plot the training data along with the boundary
view(2);
for i = 1:c
	idc = testLabel==i;
	scatter(X(1,idc),X(2,idc),36,color(mod(i-1,m)+1), 'filled');
	plot(sv(1,idc), sv(2,idc),36,color(mod(i-1,m)+1), 'black*');
end
title('Labelled Data with Support Vectors');
legend('Class 1', 'Class 2', 'Class 3','Class 4','Class 5');
figure; hold on
% Make classification predictions over a grid of values
xplot = linspace(min(features(:,1)), max(features(:,1)), 100)';
yplot = linspace(min(features(:,2)), max(features(:,2)), 100)';
[X, Y] = meshgrid(xplot, yplot);
vals = zeros(size(X));
for i = 1:size(X,2)
	x = [X(:,i),Y(:,i)];
	% Need to use evalc here to suppress LIBSVM accuracy printouts
	[T,predicted_labels, accuracy, decision_values] = evalc('svmpredict(ones(size(x(:,1))), x, model)');
	clear T;
	vals(:,i) = decision_values;
end
% Plot the SVM boundary
colormap bone
if (size(varargin, 2) == 1) && (varargin{1} == 't')
	contourf(X,Y, vals, 50, 'LineStyle', 'none');
end
	contour(X,Y, vals, [0 0], 'LineWidth', 2, 'k');
title('Labelled Data with Decision Boundary');
legend('Class 1', 'Class 2', 'Class 3','Class 4','Class 5');
end