%--------------------���������̹���--------------------
clc; clear all; clf;% ȭ���� ������ �����
fid = fopen('lenna.raw', 'r'); % ���Ͽ���
x = fread(fid,inf,'*uchar','ieee-le'); % x������ ��������
img=reshape(x,256,256); % img������ x��������̹����� �����ȯ������
figure(1); % ���α׸�
subplot(231), imshow(img');


%---------------���������̹���������׷�---------------
subplot(232), imhist(img);

%-------------���������̹�������������׷�-------------
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

%---------------------��Ȱȭ�̹���---------------------
img_eq = histeq(img,256);
subplot(234), imshow(img_eq');

%----------------��Ȱȭ�̹���������׷�-----------------
subplot(235), imhist(img_eq,256);
eqhist=imhist(img_eq);

%--------------��Ȱȭ�̹�������������׷�---------------
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
fclose(fid); % ������ ����