clc
load('C:/Users/NewUser/Desktop/ASSIGN/mnist.mat');

digits_train=double(digits_train(:,:,1:1000));
digits_test=double(digits_test(:,:,1:200));

X=zeros(1000,784);
for i=1:1000
    X(i,:)=reshape(digits_train(:,:,i),1,[]);
end
T=zeros(200,784);
for i=1:200
    T(i,:)=reshape(digits_test(:,:,i),1,[]); 
end

labels_train=double(labels_train(1:1000,1));
labels_test=double(labels_test(1:200,1));

[r,c]=size(X);

% Compute the mean of the data matrix "The mean of each row" (Equation (10))
m=mean(X')';

% Subtract the mean from each image [Centering the data]  Standardised data
d=X-repmat(m,1,c);

% Compute the covariance matrix (co) 
co=cov(d);
%co=1 / (c-1)*d*d';

% Compute the eigen values and eigen vectors of the covariance matrix 
[eigvector,eigvl]=eig(co);

% Project the original data on only two eigenvectors
PC=eigvector(:,783:784);
Data_2D=PC'*d';

figure()

for number = 0:9
    
    mask = (double(labels_train) ==  number);
    a = Data_2D(1,mask);
    b = Data_2D(2,mask);
    c = labels_train(mask);
    
    % Draw 2D visualization in separate view
%     subplot(2,5,number+1);       % Add plot in 2 x 5 grid
%     scatter(a', b');
%     title(['Number ' , num2str(number)]);
    
%Draw 2D visualization in one graph
    graph_2d(number+1) = scatter(a', b',[], c, 'filled');
    hold on;
    xlabel('PC1');
    ylabel('PC2');
    title('PCA 2D Visualization');
end

hold off;
legend([graph_2d(1),graph_2d(2),graph_2d(3),graph_2d(4),graph_2d(5),...
    graph_2d(6),graph_2d(7),graph_2d(8),graph_2d(9),graph_2d(10)],...
    'Digit 0', 'Digit 1', 'Digit 2', 'Digit 3', ...
    'Digit 4', 'Digit 5', 'Digit 6', 'Digit 7', 'Digit 8', 'Digit 9', ...
    'Location', 'Northeast', 'FontSize', 8, 'FontWeight', 'bold');


%percentage data variance

total_var=trace(eigvl);
sum_PC=eigvl(783,783)+eigvl(784,784);
percentage=(sum_PC/total_var)*100;
disp(['Percentage Data variance by first two PC=',num2str(percentage),'%'])

% Scree plot
y=zeros(1,784);
for i=1:1:784
    y(1,i)=(eigvl(i,i)/total_var)*100;
end
figure()
bar(y)
ylabel('% variance')
xlabel('Principal components')
title('Scree plot');

figure()
for number = 0:1
    
    mask = (double(labels_train) ==  number);
    a = Data_2D(1,mask);
    b = Data_2D(2,mask);
    c = labels_train(mask);
    
    % Draw 2D visualization in separate view
%     subplot(2,5,number+1);       % Add plot in 2 x 5 grid
%     scatter(a', b');
%     title(['Number ' , num2str(number)]);
    
%Draw 2D visualization in one graph
    graph_2d(number+1) = scatter(a', b',[], c, 'filled');
    hold on;
    xlabel('PC1');
    ylabel('PC2');
    title('PCA 2D Visualization');
end

hold off;
legend([graph_2d(1),graph_2d(2)],...
    'Digit 0', 'Digit 1', ...
    'Location', 'Northeast', 'FontSize', 8, 'FontWeight', 'bold');



