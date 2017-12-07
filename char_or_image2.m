function [char_to_char, image_to_char] = char_or_image2(I_mean,orthogonal_projection_matrix,max_distance,dir_name)


resize_number = 32;
number_of_element = resize_number * resize_number; %number of elements

list_test = dir(strcat(dir_name,'/*.jpg')); %getting all jpg in the folder
test_length = length(list_test); %number of test set
I_test= zeros(number_of_element, test_length); % allocate memory

for i = 1:test_length
    training_image = reshape(imresize(im2double(rgb2gray(imread(list_test(i).name))),[resize_number,resize_number]),number_of_element,1); 
    I_test(:,i) = training_image; %make image matrix
end

I_test = I_test-I_mean;
I_projected_test = orthogonal_projection_matrix * I_test;
I_test = I_test - I_projected_test;
I_test_logic = sqrt(sum(I_test.^2,1)) < max_distance;



expected_image_data = find(I_test_logic == 0);

image_to_image = 0;
for i = 1:length(expected_image_data)
    if(strcmp(list_test(expected_image_data(i)).name(1),'i'))
    image_to_image = image_to_image+1;
    end
end
character_to_image = length(expected_image_data) - image_to_image;


expected_character_data = find(I_test_logic == 1);
char_to_char = 0;
for i = 1:length(expected_character_data)
    if(strcmp(list_test(expected_character_data(i)).name(1),'c'))
    char_to_char = char_to_char+1;
    end
end
image_to_char = length(expected_character_data)-char_to_char;



end
