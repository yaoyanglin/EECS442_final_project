load('I_mean.mat');
load('max_distance.mat');
load('orthogonal_projection_matrix.mat');
addpath(genpath('ImageTest'));


list_of_folders = dir('ImageTest'); %the mother directory name
number_of_folers = length(list_of_folders)-3; %-3 for .,..,DS_STORE. IF your system is not mac, change or delete -3.

p=0; %the number of precision
e=0; %the number of error
number_of_data = 0;

for i = 1:number_of_folers
    dir_name = strcat('ImageTest/Sample',num2str(i-1));
    [acc,err] = char_or_image2(I_mean,orthogonal_projection_matrix,max_distance,dir_name);
    p = p + acc;
    e = e + err;
    number_of_data = p+e;
end



accuracy = (p/number_of_data)*100;
error = (e/number_of_data)*100;





