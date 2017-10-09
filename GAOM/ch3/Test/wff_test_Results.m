%%
%Show the computed results

result = csvread('z_filtered_1024.csv');
load('1024.mat');

colormap(gray);
subplot(1,2,1); imshow(uint8(255*mat2gray(angle(f1))));
subplot(1,2,2); imshow(uint8(255*mat2gray(angle(result))));