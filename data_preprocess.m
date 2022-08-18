% Copyright (c) 2022 Enrique Mondragon Estrada
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

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



