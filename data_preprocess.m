% Data preprocessing

close all;
clear all;
clc;

% Single image processing
m1 = imread('m1.jpg');
sz = size(m1)
imshow(m1)
[rsz_1, rszgs_1] = preprocess_single(m1);
imshowpair(rsz_1,rszgs_1,"montage")
imwrite(rszgs_1, '.\processed\m1_pp.jpg');

%%
% dataset processing

images = dir('*.jpg');      
n = length(images)

% read and show original data
for i=1:n
   name_img = images(i).name;
   curr_img = imread(name_img);
   images_org{i} = curr_img;
end
montage(images_org)

%%
% processing
images = dir('*.jpg');      
n = length(images)
for i=1:n
   name_img = images(i).name;
   curr_img = imread(name_img);
   i
   info = imfinfo(name_img);   
   if info.ColorType == "grayscale"
       rszgs_i = imresize(curr_img, [64 64]);
       images_rszgs{i} = rszgs_i;
   else
       [rsz_i, rszgs_i] = preprocess_single(curr_img);
       images_rszgs{i} = rszgs_i;
   end
end
montage(images_rszgs)

%%
% process and save the dataset
images = dir('*.jpg');      
n = length(images)
for i=1:n
   name_img = images(i).name;
   curr_img = imread(name_img);
   i
   info = imfinfo(name_img);   
   if info.ColorType == "grayscale"
       rszgs_i = imresize(curr_img, [64 64]);
       images_rszgs{i} = rszgs_i;
   else
       [rsz_i, rszgs_i] = preprocess_single(curr_img);
       images_rszgs{i} = rszgs_i;
   end
   name_img  = erase(name_img,".jpg")
   outname = sprintf('%s_pp.jpg',name_img)
   outpath =  ['.\processed\' outname]
   imwrite(im2uint8(rszgs_i), outpath);
end
montage(images_rszgs)

%%
% resize only and save
images = dir('*.jpg');      
n = length(images)
for i=1:n
   name_img = images(i).name;
   curr_img = imread(name_img);
   rszgs_i = imresize(curr_img, [64 64]);
   images_rszgs{i} = rszgs_i;
   name_img  = erase(name_img,".jpg")
   outname = sprintf('%s_pp.jpg',name_img)
   outpath =  ['.\processed_rsz\' outname]
   imwrite(im2uint8(rszgs_i), outpath);
end
montage(images_rszgs)

%%
% utils

function [rsz, rszgs] = preprocess_single(m)
    rsz = imresize(m, [64 64]);
    rszgs = 0.2989 * rsz(:, :, 1) + 0.5870 * rsz(:, :, 2) + 0.1140 * rsz(:, :, 3) ; 
end



