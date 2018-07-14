%--------------------오리지널이미지--------------------
clc; clear all; clf;% 화면을 깨끗이 만들기
fid = fopen('lenna.raw', 'r'); % 파일오픈
x = fread(fid,inf,'*uchar','ieee-le'); % x변수에 파일저장
img=reshape(x,256,256); % img변수에 x에저장된이미지를 사이즈변환후저장
figure(1); % 새로그림
subplot(231), imshow(img');


%---------------오리지널이미지히스토그램---------------
subplot(232), imhist(img);

%-------------오리지널이미지누적히스토그램-------------
img_hist = imhist(img);
histsum1 = zeros(1, 256);
for i=1:256
    histsum1(i) = 0;
end
for i=1:256
    histsum1(i+1) = histsum1(i) + img_hist(i);
end
nu_hist1 = zeros(1,256);
for i=1:256
    nu_hist1(i) = ((255.0/(256*256))*histsum1(i));
end
subplot(233), bar(nu_hist1,256);

%---------------------평활화이미지---------------------
img_eq = histeq(img,256);
subplot(234), imshow(img_eq');

%----------------평활화이미지히스토그램-----------------
subplot(235), imhist(img_eq,256);
eqhist=imhist(img_eq);

%--------------평활화이미지누적히스토그램---------------
histsum2 = zeros(1, 256);
for i=1:256
    histsum2(i) = 0;
end
for i=1:256
    histsum2(i+1) = histsum2(i) + eqhist(i);
end
nu_hist2 = zeros(1,256);
for i=1:256
    nu_hist2(i) = ((255.0/(256*256))*histsum2(i));
end
subplot(236), bar(nu_hist2,256);
fclose(fid); % 파일을 닫음