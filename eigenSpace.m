%% Basis calculation and training sample projection

%reading English's all training image
list = dir('English/Fnt/**/*.png'); %read all img files
N = size(list,1); %number of all img files
resize_number = 32;
training_number = 100;
i=1;

training_image = imresize(imread(list(i).name),[resize_number,resize_number]);%first training image
number_of_element = resize_number * resize_number; %number of elements
I = zeros(number_of_element,training_number*62); % allocate memory space


validation_number = 50;
I_validation = zeros(number_of_element, validation_number*62);

while(i <= N)
    if(strcmp(list(i).name(10:12),'101'))
        i = i + 916;
    else
        training_image = reshape(imresize(im2double(imread(list(i).name)),[resize_number,resize_number]),number_of_element,1);
        I(:,i) = training_image;
        i = i+1;
    end
end

i = 101;
j = 1;
while(i <= N)
    if(strcmp(list(i).name(10:12),'151'))
        i = i + 966;
    else
        training_image = reshape(imresize(im2double(imread(list(i).name)),[resize_number,resize_number]),number_of_element,1);
        I_validation(:,j) = training_image;
        i = i+1;
        j = j+1;
    end
end


% subtract mean
I_mean = sum(I,2)/size(I,2); %mean
I2 = I - I_mean; %normalize
% compute basis
Cov = I2*I2'; %Covariance matrix
[V,~] = eig(Cov);
r = rank(Cov);
%V = V(:,(end-r):end); %number of usable eigenvalue is 48. since most of eigenvalue is 0.
V = V(:,(end-48):end); %number of usable eigenvalue is 48. since most of eigenvalue is 0.
orthogonal_projection_matrix = V*V'; %ortho projection matrix onto subspace of eigen vectors.
% project validation samples onto subspace
I_validation = I_validation - I_mean; %normalize using training image matrix's mean
I_projected_validation = orthogonal_projection_matrix * I_validation; %orthogonal projection to eigen space
I_validation = I_validation - I_projected_validation; %difference matrix between original validation set and projection validation set
max_distance = max(sqrt(sum(I_validation.^2,1))); %getting maximum distance within validation set


% %making test-set for calculating accuracy
% i = 200;
% j = 1;
% test_number = 50;
% 
% I_test = zeros(number_of_element, test_number*62);
% 
% while(i <= N)
%     if(strcmp(list(i).name(10:12),'251'))
%         i = i + 965;
%     else
%         training_image = reshape(imresize(im2double(imread(list(i).name)),[resize_number,resize_number]),number_of_element,1);
%         I_test(:,j) = training_image;
%         i = i+1;
%         j = j+1;
%     end
% end
% 
% I_test = I_test-I_mean;
% I_projected_test = orthogonal_projection_matrix * I_test;
% I_test = I_test - I_projected_test;
% same_dataset_test_accuracy = sum(sqrt(sum(I_test.^2,1)) < max_distance)/(size(I_test,2));


%test different dataSet from Roy's
% list_test = dir('ImageTest/Sample1/*.jpg'); %read directory
% test_length = size(list_test,1); %number of test set
% I_test = zeros(number_of_element, test_length); % allocate memory
% 
% for i = 1:test_length
%     training_image = reshape(imresize(im2double(rgb2gray(imread(list_test(i).name))),[resize_number,resize_number]),number_of_element,1); 
%     I_test(:,i) = training_image; %make image matrix
% end
% 
% I_test = I_test-I_mean;
% I_projected_test = orthogonal_projection_matrix * I_test;
% I_test = I_test - I_projected_test;
% I_test_logic = sqrt(sum(I_test.^2,1)) < max_distance;
% 
% 
% 
% expected_image_data = find(I_test_logic == 0);
% 
% image_to_image = 0;
% for i = 1:length(expected_image_data)
%     if(strcmp(list_test(expected_image_data(i)).name(1),'i'))
%     image_to_image = image_to_image+1;
%     end
% end
% character_to_image = length(expected_image_data) - image_to_image;
% 
% 
% expected_character_data = find(I_test_logic == 1);
% char_to_char = 0;
% for i = 1:length(expected_character_data)
%     if(strcmp(list_test(expected_character_data(i)).name(1),'c'))
%     char_to_char = char_to_char+1;
%     end
% end
% image_to_char = length(expected_character_data)-char_to_char;
% 















